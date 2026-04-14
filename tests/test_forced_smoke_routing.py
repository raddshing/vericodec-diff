from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.cells import build_cell_schema
from rd_lora.training import execution
from rd_lora.training.adapter_factory import build_backend_adapter_plan
from rd_lora.training.timestep_routing import (
    build_timestep_band_routes,
    project_training_timestep_to_step_index,
    resolve_deterministic_timestep_for_band,
    resolve_timestep_band_name_for_training_timesteps,
)


def _timestep_only_manifest(*, zero_rank_bands: set[str] | None = None) -> dict[str, object]:
    zero_rank_bands = zero_rank_bands or set()
    schema = build_cell_schema()
    cells = []
    for cell in schema.cells:
        timestep_band = cell.timestep_band.band_id
        rank = 0 if timestep_band in zero_rank_bands else 4
        cells.append(
            {
                "cell_id": cell.cell_id,
                "layer_group": cell.layer_group.group_id,
                "timestep_band": timestep_band,
                "rank": rank,
                "alpha": rank,
                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
                "adapter_name": f"timestep_only_{timestep_band}",
            }
        )
    return {
        "schema_version": "1.0",
        "backend": "timestep_only",
        "rank_budget_total": sum(int(cell["rank"]) for cell in cells),
        "layer_groups": [group.group_id for group in schema.layer_groups],
        "timestep_bands": [band.band_id for band in schema.timestep_bands],
        "cells": cells,
    }


def _adapter_specs_for_plan(plan: dict[str, object]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for bank in plan["adapter_banks"]:
        specs.append(
            {
                "adapter_name": bank["adapter_name"],
                "timestep_band": bank["timestep_band"],
                "cell_ids": list(bank["cell_ids"]),
                "target_modules": ["down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"],
                "rank": int(bank.get("rank", 4)),
                "alpha": int(bank.get("alpha", bank.get("rank", 4))),
                "rank_pattern": {},
                "alpha_pattern": {},
                "noop": False,
            }
        )
    return specs


def _training_args(output_dir: Path, *, max_train_steps: int) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=str(output_dir),
        logging_dir=str((output_dir / "logs").resolve()),
        pretrained_model_name_or_path="stub-model",
        pretrained_vae_model_name_or_path=None,
        revision=None,
        variant=None,
        gradient_accumulation_steps=1,
        mixed_precision="fp32",
        report_to="none",
        lora_dropout=0.0,
        learning_rate=1.0e-4,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_weight_decay=1.0e-4,
        adam_epsilon=1.0e-8,
        max_grad_norm=1.0,
        max_train_steps=max_train_steps,
        checkpointing_steps=100,
    )


def _provenance(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "run_mode": "real_gpu",
        "used_gpu": True,
        "python": "3.11.0",
        "torch": "2.5.0",
        "torch_cuda_is_available": True,
        "torch_cuda_version": "12.1",
        "torch_device_count": 1,
        "gpu_names": ["Fake GPU 0"],
        "diffusers": "0.0.test",
        "diffusers_file": str((tmp_path / "fake_diffusers.py").resolve()),
        "accelerate_config_file": str((tmp_path / "accelerate.yaml").resolve()),
        "git_commit": "abc123",
        "backend": "timestep_only",
        "task": "subject_personalization",
        "allocation_manifest": str((tmp_path / "allocation.json").resolve()),
        "peak_vram_mib": 512.0,
        "timestamp_utc": "2026-04-14T00:00:00Z",
    }


class FakeTensor:
    def __init__(self, values: list[int]) -> None:
        self._values = list(values)

    def long(self) -> "FakeTensor":
        return self

    def detach(self) -> "FakeTensor":
        return self

    def flatten(self) -> "FakeTensor":
        return self

    def tolist(self) -> list[int]:
        return list(self._values)


class FakePixelValues:
    def __init__(self, *, batch_size: int) -> None:
        self.shape = (batch_size, 4, 64, 64)
        self.device = "cuda:0"


class FakeParam:
    def __init__(self, *, requires_grad: bool = False, name: str = "") -> None:
        self.requires_grad = requires_grad
        self.grad = None
        self.name = name

    def numel(self) -> int:
        return 1


class FakeModel:
    def __init__(self, *, parameter_count: int = 0) -> None:
        self._parameters = [FakeParam(requires_grad=False, name=f"base_param_{i}") for i in range(parameter_count)]
        self.adapter_params: dict[str, list[FakeParam]] = {}
        self.active_adapter = ""
        self.adapters_enabled = True
        self.gradient_checkpointing = False

    def register_adapter_params(self, adapter_name: str, *, count: int = 1) -> list[FakeParam]:
        params = [
            FakeParam(
                requires_grad=True,
                name=f"unet.to_q.{adapter_name}.lora_{index}.weight",
            )
            for index in range(count)
        ]
        self.adapter_params[adapter_name] = params
        self._parameters.extend(params)
        return params

    def parameters(self):  # noqa: ANN201
        return list(self._parameters)

    def named_parameters(self):  # noqa: ANN201
        return [(parameter.name, parameter) for parameter in self._parameters]

    def named_modules(self):  # noqa: ANN201
        return []

    def requires_grad_(self, _enabled: bool) -> "FakeModel":
        return self

    def to(self, *args, **kwargs) -> "FakeModel":  # noqa: ANN002, ANN003
        return self

    def train(self) -> None:
        return None

    def set_adapter(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name

    def enable_adapters(self) -> None:
        self.adapters_enabled = True

    def disable_adapters(self) -> None:
        self.adapters_enabled = False

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True


class FakeLoss:
    def __init__(self, value: float) -> None:
        self.value = value

    def detach(self) -> "FakeLoss":
        return self

    def item(self) -> float:
        return self.value


class FakeOptimizer:
    def __init__(self, parameters) -> None:  # noqa: ANN001
        self.parameters = list(parameters)
        self.param_groups = [{"params": list(self.parameters)}]
        self.step_calls = 0
        self.zero_grad_calls = 0

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.zero_grad_calls += 1
        for parameter in self.parameters:
            parameter.grad = None if set_to_none else 0.0

    def step(self) -> None:
        self.step_calls += 1


class FakeLRScheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1

    def get_last_lr(self) -> list[float]:
        return [1.0e-4]


class FakeNoiseScheduler:
    def __init__(self) -> None:
        self.config = type("Config", (), {"num_train_timesteps": 1000, "prediction_type": "epsilon"})()


class FakeAccelerator:
    def __init__(self, *, model: FakeModel | None = None) -> None:
        self.device = "cuda:0"
        self.mixed_precision = "fp32"
        self.num_processes = 1
        self.sync_gradients = True
        self.is_main_process = True
        self.model = model
        self.optimizer: FakeOptimizer | None = None

    def prepare(self, *items):  # noqa: ANN002
        return items

    def accumulate(self, _model):  # noqa: ANN001
        return nullcontext()

    def backward(self, _loss: FakeLoss) -> None:
        if self.model is None or self.optimizer is None or not self.model.adapters_enabled:
            return None
        for parameter in self.model.adapter_params.get(self.model.active_adapter, []):
            parameter.grad = 1.0
        return None

    def clip_grad_norm_(self, _parameters, _max_grad_norm) -> None:  # noqa: ANN001
        return None

    def log(self, _payload, step: int) -> None:  # noqa: ANN001
        return None

    def save_state(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def init_trackers(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
        return None


class FakeCuda:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def reset_peak_memory_stats(self) -> None:
        return None

    def max_memory_allocated(self) -> int:
        return 512 * 1024 * 1024


class FakeTorch:
    def __init__(self) -> None:
        self.cuda = FakeCuda()
        self.float16 = "float16"
        self.bfloat16 = "bfloat16"
        self.float32 = "float32"
        self.long = "long"

    def full(self, shape, value: int, **_kwargs) -> FakeTensor:  # noqa: ANN001
        batch_size = int(shape[0])
        return FakeTensor([int(value)] * batch_size)


def test_resolve_deterministic_timestep_for_band_round_trips_all_schema_bands() -> None:
    schema = build_cell_schema()
    routing_table = build_timestep_band_routes(
        [band.band_id for band in schema.timestep_bands],
        schema=schema,
    )

    for band in schema.timestep_bands:
        timestep_value = resolve_deterministic_timestep_for_band(routing_table, band.band_id)
        resolved_band = resolve_timestep_band_name_for_training_timesteps(
            routing_table,
            timesteps=[timestep_value],
            num_train_timesteps=1000,
        )
        projected_step_index = project_training_timestep_to_step_index(
            timestep_value,
            num_train_timesteps=1000,
            reference_step_count=20,
        )

        assert resolved_band == band.band_id
        assert projected_step_index in set(band.step_indices)


def test_execute_training_run_forced_sequence_routes_noop_then_trainable(monkeypatch, tmp_path: Path) -> None:
    plan = build_backend_adapter_plan(
        "timestep_only",
        _timestep_only_manifest(zero_rank_bands={"timestep_band_00", "timestep_band_01"}),
    )
    adapter_specs = _adapter_specs_for_plan(plan)
    fake_torch = FakeTorch()
    fake_unet = FakeModel()
    fake_accelerator = FakeAccelerator(model=fake_unet)
    fake_lr_scheduler = FakeLRScheduler()
    fake_components = {
        "module": type("Module", (), {"cast_training_params": staticmethod(lambda *_a, **_k: None)})(),
        "tokenizer": object(),
        "tokenizer_2": object(),
        "text_encoder": FakeModel(parameter_count=0),
        "text_encoder_2": FakeModel(parameter_count=0),
        "vae": FakeModel(parameter_count=0),
        "unet": fake_unet,
        "scheduler": FakeNoiseScheduler(),
    }
    optimizer_holder: dict[str, FakeOptimizer] = {}
    planned_route_bands: list[str] = []
    loss_route_bands: list[str] = []
    original_plan_route = execution._plan_timestep_band_batch_route

    monkeypatch.setattr(execution, "load_allocation_manifest", lambda _path: {"backend": "timestep_only"})
    monkeypatch.setattr(
        execution,
        "build_training_args",
        lambda **_kwargs: _training_args(tmp_path, max_train_steps=5),
    )
    monkeypatch.setattr(
        execution,
        "validate_official_sdxl_dreambooth_script",
        lambda *_args, **_kwargs: (tmp_path / "train_dreambooth_lora_sdxl.py").resolve(),
    )
    monkeypatch.setattr(execution, "load_torch", lambda: fake_torch)
    monkeypatch.setattr(execution, "create_accelerator", lambda _args: fake_accelerator)
    monkeypatch.setattr(execution, "load_official_sdxl_components", lambda **_kwargs: fake_components)
    monkeypatch.setattr(
        execution,
        "_configure_components_for_training",
        lambda **_kwargs: {"weight_dtype": "float32", "latents_mean": None, "latents_std": None},
    )
    monkeypatch.setattr(execution, "build_backend_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(execution, "build_concrete_adapter_specs", lambda _plan, unet: adapter_specs)

    def fake_apply_adapter_specs(**kwargs) -> list[str]:  # noqa: ANN003
        for spec in kwargs["adapter_specs"]:
            kwargs["unet"].register_adapter_params(str(spec["adapter_name"]))
        return [str(spec["adapter_name"]) for spec in kwargs["adapter_specs"]]

    monkeypatch.setattr(execution, "apply_adapter_specs", fake_apply_adapter_specs)
    monkeypatch.setattr(
        execution,
        "create_train_dataloader",
        lambda **_kwargs: [
            {"pixel_values": FakePixelValues(batch_size=1)},
            {"pixel_values": FakePixelValues(batch_size=1)},
            {"pixel_values": FakePixelValues(batch_size=1)},
        ],
    )

    def fake_create_optimizer(*, trainable_parameters, **_kwargs):  # noqa: ANN003
        optimizer = FakeOptimizer(trainable_parameters)
        optimizer_holder["optimizer"] = optimizer
        fake_accelerator.optimizer = optimizer
        return optimizer

    monkeypatch.setattr(execution, "create_optimizer", fake_create_optimizer)
    monkeypatch.setattr(execution, "create_lr_scheduler", lambda **_kwargs: fake_lr_scheduler)

    def recording_plan_route(**kwargs):  # noqa: ANN003
        route = original_plan_route(**kwargs)
        planned_route_bands.append(str(route["active_timestep_band"]))
        return route

    monkeypatch.setattr(execution, "_plan_timestep_band_batch_route", recording_plan_route)

    def fake_compute_step_loss(**kwargs):  # noqa: ANN003
        loss_route_bands.append(str(kwargs["active_timestep_band"]))
        assert str(kwargs["active_timestep_band"]) == "timestep_band_02"
        assert str(kwargs["active_adapter_name"]) == "timestep_only_timestep_band_02"
        assert kwargs["timesteps"].tolist()
        return FakeLoss(2.0)

    monkeypatch.setattr(execution, "compute_step_loss", fake_compute_step_loss)
    monkeypatch.setattr(execution, "build_run_provenance_payload", lambda **_kwargs: _provenance(tmp_path))

    def fake_save_checkpoint(*, accelerator, output_dir, global_step):  # noqa: ANN001
        checkpoint_dir = output_dir / f"checkpoint-{global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    monkeypatch.setattr(execution, "save_checkpoint", fake_save_checkpoint)

    result = execution.execute_training_run(
        repo_root=REPO_ROOT,
        config={},
        task="subject_personalization",
        backend="timestep_only",
        allocation_path=tmp_path / "allocation.json",
        output_dir=tmp_path,
        run_mode="real_gpu",
        forced_timestep_band_sequence=["timestep_band_00", "timestep_band_02"],
    )

    assert planned_route_bands == ["timestep_band_00", "timestep_band_02"]
    assert loss_route_bands == ["timestep_band_02"]
    assert optimizer_holder["optimizer"].step_calls == 1
    assert fake_lr_scheduler.step_calls == 1
    assert result["metrics"]["skipped_noop_band_batches"] == 1
    assert result["metrics"]["active_band_optimizer_steps"] == 1
    assert result["train_summary"]["train_steps"] == 1
