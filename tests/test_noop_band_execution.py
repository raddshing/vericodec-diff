from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.cells import build_cell_schema
from rd_lora.training import adapter_factory, execution
from rd_lora.training.adapter_factory import build_backend_adapter_plan


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


def _proposed_partial_sparse_manifest(*, partial_sparse_band: str) -> dict[str, object]:
    schema = build_cell_schema()
    cells = []
    for cell in schema.cells:
        timestep_band = cell.timestep_band.band_id
        if timestep_band == partial_sparse_band:
            rank = 4 if (cell.layer_group.group_index % 2) == 0 else 0
        else:
            rank = 0
        cells.append(
            {
                "cell_id": cell.cell_id,
                "layer_group": cell.layer_group.group_id,
                "timestep_band": timestep_band,
                "rank": rank,
                "alpha": rank,
                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
                "adapter_name": f"proposed_{timestep_band}",
            }
        )
    return {
        "schema_version": "1.0",
        "backend": "proposed",
        "rank_budget_total": sum(int(cell["rank"]) for cell in cells),
        "layer_groups": [group.group_id for group in schema.layer_groups],
        "timestep_bands": [band.band_id for band in schema.timestep_bands],
        "cells": cells,
    }


class FakeParam:
    def __init__(self, *, requires_grad: bool = False) -> None:
        self.requires_grad = requires_grad
        self.grad = None


class FakeModel:
    def __init__(self, *, parameter_count: int = 0) -> None:
        self._parameters = [FakeParam(requires_grad=False) for _ in range(parameter_count)]
        self.adapter_params: dict[str, list[FakeParam]] = {}
        self.active_adapter = ""
        self.adapters_enabled = True
        self.set_adapter_calls: list[str] = []
        self.enable_adapter_calls = 0
        self.disable_adapter_calls = 0

    def register_adapter_params(self, adapter_name: str, *, count: int = 1) -> list[FakeParam]:
        params = [FakeParam(requires_grad=True) for _ in range(count)]
        self.adapter_params[adapter_name] = params
        self._parameters.extend(params)
        return params

    def parameters(self):  # noqa: ANN201
        return list(self._parameters)

    def named_modules(self):  # noqa: ANN201
        return []

    def requires_grad_(self, _enabled: bool) -> "FakeModel":
        return self

    def to(self, *args, **kwargs) -> "FakeModel":  # noqa: ANN002, ANN003
        return self

    def train(self) -> None:
        return None

    def add_adapter(self, _adapter_config, adapter_name: str = "default") -> None:  # noqa: ANN001
        self.active_adapter = adapter_name

    def set_adapter(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name
        self.set_adapter_calls.append(adapter_name)

    def enable_adapters(self) -> None:
        self.adapters_enabled = True
        self.enable_adapter_calls += 1

    def disable_adapters(self) -> None:
        self.adapters_enabled = False
        self.disable_adapter_calls += 1


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
        self.zero_grad_calls = 0
        self.step_calls = 0
        self.step_grad_counts: list[int] = []

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.zero_grad_calls += 1
        for parameter in self.parameters:
            parameter.grad = None if set_to_none else 0.0

    def step(self) -> None:
        self.step_calls += 1
        self.step_grad_counts.append(sum(1 for parameter in self.parameters if parameter.grad is not None))


class FakeScheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1

    def get_last_lr(self) -> list[float]:
        return [1.0e-4]


class FakeAccelerator:
    def __init__(self, *, model: FakeModel | None = None) -> None:
        self.device = "cuda:0"
        self.mixed_precision = "fp32"
        self.num_processes = 1
        self.sync_gradients = True
        self.is_main_process = True
        self.model = model
        self.optimizer: FakeOptimizer | None = None
        self.backward_calls: list[FakeLoss] = []
        self.grad_param_counts: list[int] = []
        self.logged_steps: list[int] = []
        self.accumulate_calls = 0
        self.accumulate_entry_adapters: list[str] = []
        self.accumulate_entry_adapter_states: list[bool] = []
        self.accumulate_entry_timestep_bands: list[str] = []
        self.planned_route_bands: list[str] = []
        self.loss_route_bands: list[str] = []

    def prepare(self, *items):  # noqa: ANN002
        return items

    def accumulate(self, model):  # noqa: ANN001
        self.accumulate_calls += 1
        self.accumulate_entry_adapters.append(str(getattr(model, "active_adapter", "")))
        self.accumulate_entry_adapter_states.append(bool(getattr(model, "adapters_enabled", True)))
        self.accumulate_entry_timestep_bands.append(str(getattr(model, "_rd_lora_active_timestep_band", "")))
        return nullcontext()

    def backward(self, loss: FakeLoss) -> None:
        self.backward_calls.append(loss)
        grad_count = 0
        if self.model is not None and self.optimizer is not None and self.model.adapters_enabled:
            for parameter in self.model.adapter_params.get(self.model.active_adapter, []):
                parameter.grad = 1.0
                grad_count += 1
        self.grad_param_counts.append(grad_count)

    def clip_grad_norm_(self, _parameters, _max_grad_norm) -> None:  # noqa: ANN001
        return None

    def log(self, _payload, step: int) -> None:  # noqa: ANN001
        self.logged_steps.append(step)

    def save_state(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def init_trackers(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
        return None


class FakeCuda:
    def __init__(self) -> None:
        self.reset_called = False

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def reset_peak_memory_stats(self) -> None:
        self.reset_called = True

    def max_memory_allocated(self) -> int:
        return 512 * 1024 * 1024


class FakeTorch:
    def __init__(self) -> None:
        self.cuda = FakeCuda()
        self.float16 = "float16"
        self.bfloat16 = "bfloat16"
        self.float32 = "float32"


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
        "timestamp_utc": "2026-04-11T00:00:00Z",
    }


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


def _fake_validate_cell_targets(schema, _unet, _target_modules):  # noqa: ANN001
    return {
        cell.cell_id: [f"{cell.cell_id}.to_q"]
        for cell in schema.cells
    }


def _optimizer_param_ids(optimizer: FakeOptimizer) -> list[int]:
    return [id(parameter) for parameter in optimizer.parameters]


def _run_training_scenario(
    monkeypatch,
    tmp_path: Path,
    *,
    routed_bands: list[str],
    max_train_steps: int,
    plan: dict[str, object] | None = None,
) -> tuple[dict[str, object], FakeAccelerator, FakeOptimizer, FakeScheduler, FakeModel, list[dict[str, object]]]:
    plan = plan or build_backend_adapter_plan(
        "timestep_only",
        _timestep_only_manifest(zero_rank_bands={"timestep_band_00"}),
    )
    adapter_specs = _adapter_specs_for_plan(plan)
    fake_torch = FakeTorch()
    fake_unet = FakeModel()
    fake_components = {
        "module": type("Module", (), {"cast_training_params": staticmethod(lambda *_a, **_k: None)})(),
        "tokenizer": object(),
        "tokenizer_2": object(),
        "text_encoder": FakeModel(parameter_count=0),
        "text_encoder_2": FakeModel(parameter_count=0),
        "vae": FakeModel(parameter_count=0),
        "unet": fake_unet,
        "scheduler": object(),
    }
    fake_accelerator = FakeAccelerator(model=fake_unet)
    fake_scheduler = FakeScheduler()
    optimizer_holder: dict[str, FakeOptimizer] = {}
    schedule = iter(routed_bands)

    monkeypatch.setattr(execution, "load_allocation_manifest", lambda _path: {"backend": "timestep_only"})
    monkeypatch.setattr(execution, "build_training_args", lambda **_kwargs: _training_args(tmp_path, max_train_steps=max_train_steps))
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

    def fake_apply_adapter_specs(**kwargs) -> None:  # noqa: ANN003
        for spec in kwargs["adapter_specs"]:
            kwargs["unet"].register_adapter_params(str(spec["adapter_name"]))

    monkeypatch.setattr(execution, "apply_adapter_specs", fake_apply_adapter_specs)
    monkeypatch.setattr(execution, "create_train_dataloader", lambda **_kwargs: [{"batch": 1}])

    def fake_create_optimizer(*, trainable_parameters, **_kwargs):  # noqa: ANN003
        optimizer = FakeOptimizer(trainable_parameters)
        optimizer_holder["optimizer"] = optimizer
        fake_accelerator.optimizer = optimizer
        return optimizer

    monkeypatch.setattr(execution, "create_optimizer", fake_create_optimizer)
    monkeypatch.setattr(execution, "create_lr_scheduler", lambda **_kwargs: fake_scheduler)

    def fake_plan_timestep_band_batch_route(**kwargs):  # noqa: ANN003
        try:
            timestep_band = next(schedule)
        except StopIteration as exc:
            raise AssertionError("timestep-band route planning called more times than expected") from exc
        fake_accelerator.planned_route_bands.append(timestep_band)
        route = kwargs["plan"]["routing"]["table"][timestep_band]
        return {
            "timesteps": object(),
            "active_timestep_band": timestep_band,
            "adapter_name": route["adapter_name"],
            "noop": bool(route["noop"]),
        }

    monkeypatch.setattr(execution, "_plan_timestep_band_batch_route", fake_plan_timestep_band_batch_route)

    def fake_compute_step_loss(**kwargs):  # noqa: ANN003
        timestep_band = str(kwargs["active_timestep_band"])
        fake_accelerator.loss_route_bands.append(timestep_band)
        assert kwargs["timesteps"] is not None
        assert kwargs["active_adapter_name"] == kwargs["plan"]["routing"]["table"][timestep_band]["adapter_name"]
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
    )
    return result, fake_accelerator, optimizer_holder["optimizer"], fake_scheduler, fake_unet, adapter_specs


def test_backend_plan_routes_noop_band_to_none() -> None:
    plan = build_backend_adapter_plan(
        "timestep_only",
        _timestep_only_manifest(zero_rank_bands={"timestep_band_00"}),
    )

    assert plan["timestep_band_metadata"]["timestep_band_00"]["has_trainable_params"] is False
    assert plan["timestep_band_metadata"]["timestep_band_00"]["noop"] is True
    assert plan["routing"]["table"]["timestep_band_00"]["noop"] is True
    assert plan["routing"]["table"]["timestep_band_00"]["adapter_name"] is None
    assert plan["timestep_band_metadata"]["timestep_band_01"]["has_trainable_params"] is True
    assert plan["timestep_band_metadata"]["timestep_band_01"]["noop"] is False
    assert plan["routing"]["table"]["timestep_band_01"]["adapter_name"] == "timestep_only_timestep_band_01"


def test_noop_to_trainable_transition_records_optimizer_grads(monkeypatch, tmp_path: Path) -> None:
    result, accelerator, optimizer, scheduler, unet, _adapter_specs = _run_training_scenario(
        monkeypatch,
        tmp_path,
        routed_bands=["timestep_band_00", "timestep_band_01"],
        max_train_steps=1,
    )

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert len(accelerator.backward_calls) == 1
    assert accelerator.grad_param_counts == [1]
    assert accelerator.accumulate_calls == 1
    assert accelerator.planned_route_bands == ["timestep_band_00", "timestep_band_01"]
    assert accelerator.loss_route_bands == ["timestep_band_01"]
    assert accelerator.accumulate_entry_timestep_bands == ["timestep_band_01"]
    assert accelerator.accumulate_entry_adapter_states == [True]
    assert optimizer.step_calls == 1
    assert optimizer.step_grad_counts == [1]
    assert optimizer.zero_grad_calls == 1
    assert scheduler.step_calls == 1
    assert unet.disable_adapter_calls == 0
    assert unet.enable_adapter_calls == 1
    assert unet.set_adapter_calls == ["timestep_only_timestep_band_01"]
    assert metrics["skipped_noop_band_batches"] == 1
    assert metrics["skipped_noop_band_fraction"] == 0.5
    assert metrics["active_band_optimizer_steps"] == metrics["global_step"] == 1
    assert result["metrics"] == metrics


def test_proposed_partial_sparse_band_builds_trainable_spec(monkeypatch) -> None:
    partial_sparse_band = "timestep_band_01"
    plan = build_backend_adapter_plan(
        "proposed",
        _proposed_partial_sparse_manifest(partial_sparse_band=partial_sparse_band),
    )

    assert plan["timestep_band_metadata"][partial_sparse_band]["noop"] is False
    assert plan["routing"]["table"][partial_sparse_band]["noop"] is False
    assert plan["routing"]["table"][partial_sparse_band]["adapter_name"] == f"proposed_{partial_sparse_band}"
    assert plan["routing"]["table"]["timestep_band_00"]["adapter_name"] is None

    monkeypatch.setattr(adapter_factory, "validate_cell_targets", _fake_validate_cell_targets)
    specs = adapter_factory.build_concrete_adapter_specs(plan, unet=FakeModel(parameter_count=0))
    spec_by_name = {str(spec["adapter_name"]): spec for spec in specs}

    assert f"proposed_{partial_sparse_band}" in spec_by_name
    assert spec_by_name[f"proposed_{partial_sparse_band}"]["noop"] is False
    assert spec_by_name[f"proposed_{partial_sparse_band}"]["target_modules"]
    assert "proposed_timestep_band_00" not in spec_by_name


def test_optimizer_coverage_matches_precreated_trainable_adapters(monkeypatch, tmp_path: Path) -> None:
    _result, _accelerator, optimizer, _scheduler, unet, adapter_specs = _run_training_scenario(
        monkeypatch,
        tmp_path,
        routed_bands=["timestep_band_01"],
        max_train_steps=1,
    )

    expected_adapter_names = {str(spec["adapter_name"]) for spec in adapter_specs}
    assert set(unet.adapter_params) == expected_adapter_names

    optimizer_param_ids = _optimizer_param_ids(optimizer)
    expected_param_ids = [
        id(parameter)
        for adapter_name in sorted(expected_adapter_names)
        for parameter in unet.adapter_params[adapter_name]
    ]
    assert sorted(optimizer_param_ids) == sorted(expected_param_ids)
    assert len(optimizer_param_ids) == len(set(optimizer_param_ids))


def test_no_noop_spec_in_adapter_spec_lookup(monkeypatch) -> None:
    plan = build_backend_adapter_plan(
        "timestep_only",
        _timestep_only_manifest(zero_rank_bands={"timestep_band_00"}),
    )

    monkeypatch.setattr(adapter_factory, "validate_cell_targets", _fake_validate_cell_targets)
    adapter_specs = adapter_factory.build_concrete_adapter_specs(plan, unet=FakeModel(parameter_count=0))
    adapter_spec_by_name = execution._adapter_spec_lookup(adapter_specs)

    assert "timestep_only_timestep_band_00" not in adapter_spec_by_name
    assert adapter_spec_by_name["timestep_only_timestep_band_01"]["noop"] is False
    assert all(spec["noop"] is False for spec in adapter_spec_by_name.values())
