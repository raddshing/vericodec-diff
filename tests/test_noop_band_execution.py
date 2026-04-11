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
from rd_lora.training import execution
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


class FakeParam:
    def __init__(self, *, requires_grad: bool = False) -> None:
        self.requires_grad = requires_grad


class FakeModel:
    def __init__(self, *, parameter_count: int = 2) -> None:
        self._parameters = [FakeParam(requires_grad=False) for _ in range(parameter_count)]
        self.active_adapter = ""
        self.adapters_enabled = True

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

    def enable_adapters(self) -> None:
        self.adapters_enabled = True

    def disable_adapters(self) -> None:
        self.adapters_enabled = False


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
        self.zero_grad_calls = 0
        self.step_calls = 0

    def zero_grad(self) -> None:
        self.zero_grad_calls += 1

    def step(self) -> None:
        self.step_calls += 1


class FakeScheduler:
    def __init__(self) -> None:
        self.step_calls = 0

    def step(self) -> None:
        self.step_calls += 1

    def get_last_lr(self) -> list[float]:
        return [1.0e-4]


class FakeAccelerator:
    def __init__(self) -> None:
        self.device = "cuda:0"
        self.mixed_precision = "fp32"
        self.num_processes = 1
        self.sync_gradients = True
        self.is_main_process = True
        self.backward_calls: list[FakeLoss] = []
        self.logged_steps: list[int] = []

    def prepare(self, *items):  # noqa: ANN002
        return items

    def accumulate(self, _model):  # noqa: ANN001
        return nullcontext()

    def backward(self, loss: FakeLoss) -> None:
        self.backward_calls.append(loss)

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
    routing_table = plan["routing"]["table"]
    metadata = plan["timestep_band_metadata"]
    specs: list[dict[str, object]] = []
    for timestep_band in plan["timestep_bands"]:
        is_noop = bool(metadata[timestep_band]["noop"])
        specs.append(
            {
                "adapter_name": routing_table[timestep_band]["adapter_name"],
                "timestep_band": timestep_band,
                "cell_ids": list(metadata[timestep_band]["cell_ids"]),
                "target_modules": [] if is_noop else ["down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"],
                "rank": 0 if is_noop else 4,
                "alpha": 0 if is_noop else 4,
                "rank_pattern": {},
                "alpha_pattern": {},
                "noop": is_noop,
            }
        )
    return specs


def _run_training_scenario(
    monkeypatch,
    tmp_path: Path,
    *,
    routed_bands: list[str],
    max_train_steps: int,
) -> tuple[dict[str, object], FakeAccelerator, FakeOptimizer, FakeScheduler]:
    plan = build_backend_adapter_plan(
        "timestep_only",
        _timestep_only_manifest(zero_rank_bands={"timestep_band_00"}),
    )
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
    fake_accelerator = FakeAccelerator()
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
    monkeypatch.setattr(execution, "build_concrete_adapter_specs", lambda _plan, unet: _adapter_specs_for_plan(_plan))

    def fake_apply_adapter_specs(**kwargs) -> None:  # noqa: ANN003
        if any(not bool(spec["noop"]) for spec in kwargs["adapter_specs"]):
            kwargs["unet"].parameters()[0].requires_grad = True

    monkeypatch.setattr(execution, "apply_adapter_specs", fake_apply_adapter_specs)
    monkeypatch.setattr(execution, "create_train_dataloader", lambda **_kwargs: [{"batch": 1}])

    def fake_create_optimizer(*, trainable_parameters, **_kwargs):  # noqa: ANN003
        optimizer = FakeOptimizer(trainable_parameters)
        optimizer_holder["optimizer"] = optimizer
        return optimizer

    monkeypatch.setattr(execution, "create_optimizer", fake_create_optimizer)
    monkeypatch.setattr(execution, "create_lr_scheduler", lambda **_kwargs: fake_scheduler)

    def fake_compute_step_loss(**kwargs):  # noqa: ANN003
        try:
            timestep_band = next(schedule)
        except StopIteration as exc:
            raise AssertionError("compute_step_loss called more times than expected") from exc
        adapter_name = kwargs["plan"]["routing"]["table"][timestep_band]["adapter_name"]
        kwargs["components"]["unet"].set_adapter(adapter_name)
        kwargs["components"]["unet"]._rd_lora_active_timestep_band = timestep_band
        return FakeLoss(1.0 if timestep_band == "timestep_band_00" else 2.0), adapter_name

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
    return result, fake_accelerator, optimizer_holder["optimizer"], fake_scheduler


def test_backend_plan_exposes_noop_timestep_band_metadata() -> None:
    plan = build_backend_adapter_plan(
        "timestep_only",
        _timestep_only_manifest(zero_rank_bands={"timestep_band_00"}),
    )

    assert plan["timestep_band_metadata"]["timestep_band_00"]["has_trainable_params"] is False
    assert plan["timestep_band_metadata"]["timestep_band_00"]["noop"] is True
    assert plan["timestep_band_metadata"]["timestep_band_01"]["has_trainable_params"] is True
    assert plan["timestep_band_metadata"]["timestep_band_01"]["noop"] is False


def test_zero_rank_timestep_band_does_not_call_backward_or_step(monkeypatch, tmp_path: Path) -> None:
    result, accelerator, optimizer, scheduler = _run_training_scenario(
        monkeypatch,
        tmp_path,
        routed_bands=["timestep_band_00", "timestep_band_01"],
        max_train_steps=1,
    )

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert len(accelerator.backward_calls) == 1
    assert optimizer.step_calls == 1
    assert scheduler.step_calls == 1
    assert metrics["skipped_noop_band_batches"] == 1
    assert metrics["skipped_noop_band_fraction"] == 0.5
    assert metrics["active_band_optimizer_steps"] == metrics["global_step"] == 1
    assert result["metrics"] == metrics


def test_trainable_band_still_calls_backward_and_step(monkeypatch, tmp_path: Path) -> None:
    result, accelerator, optimizer, scheduler = _run_training_scenario(
        monkeypatch,
        tmp_path,
        routed_bands=["timestep_band_01"],
        max_train_steps=1,
    )

    assert len(accelerator.backward_calls) == 1
    assert optimizer.step_calls == 1
    assert scheduler.step_calls == 1
    assert result["metrics"]["skipped_noop_band_batches"] == 0
    assert result["metrics"]["active_band_optimizer_steps"] == 1


def test_global_step_counts_only_optimizer_steps_not_raw_batches(monkeypatch, tmp_path: Path) -> None:
    result, accelerator, optimizer, scheduler = _run_training_scenario(
        monkeypatch,
        tmp_path,
        routed_bands=[
            "timestep_band_00",
            "timestep_band_01",
            "timestep_band_00",
            "timestep_band_01",
        ],
        max_train_steps=2,
    )

    assert optimizer.zero_grad_calls == 4
    assert len(accelerator.backward_calls) == 2
    assert optimizer.step_calls == 2
    assert scheduler.step_calls == 2
    assert accelerator.logged_steps == [1, 2]
    assert result["metrics"]["global_step"] == 2
    assert result["metrics"]["active_band_optimizer_steps"] == 2
    assert result["metrics"]["skipped_noop_band_batches"] == 2
    assert result["metrics"]["skipped_noop_band_fraction"] == 0.5
