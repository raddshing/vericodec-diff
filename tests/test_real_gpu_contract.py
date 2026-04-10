from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.training import execution


MANIFEST_PATH = REPO_ROOT / "outputs" / "rd_lora" / "allocation" / "gate_real" / "uniform.json"


def _config() -> dict[str, object]:
    return {
        "paths": {
            "repo_root": ".",
            "official_diffusers_script": "baselines/external/huggingface_diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py",
            "accelerate_config": "configs/accelerate/single_gpu_fp16.yaml",
            "expected_diffusers_substring": "huggingface_diffusers",
        },
        "model": {
            "pretrained_model_name_or_path": "stub-model",
        },
        "training": {
            "resolution": 1024,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_train_steps": 1,
            "checkpointing_steps": 1,
            "checkpoints_total_limit": 1,
            "learning_rate": 1.0e-4,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "lr_num_cycles": 1,
            "lr_power": 1.0,
            "report_to": "tensorboard",
            "mixed_precision": "fp16",
            "dataloader_num_workers": 0,
            "num_validation_images": 1,
            "validation_epochs": 1000,
            "center_crop": True,
            "gradient_checkpointing": True,
            "optimizer": "AdamW",
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_weight_decay": 1.0e-4,
            "adam_weight_decay_text_encoder": 1.0e-3,
            "adam_epsilon": 1.0e-8,
            "max_grad_norm": 1.0,
            "lora_dropout": 0.0,
            "repeats": 1,
            "seed": 123,
            "use_rslora": True,
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        },
        "tasks": {
            "subject_personalization": {
                "train_data_dir": "tests/fixtures/rdlora_pilot",
                "instance_prompt": "a studio portrait of the subject",
                "validation_prompt": "A studio portrait of the same subject.",
            }
        },
    }


class _Param:
    def __init__(self) -> None:
        self.requires_grad = False


class _Model:
    def __init__(self, *, parameter_count: int = 1) -> None:
        self._parameters = [_Param() for _ in range(parameter_count)]

    def parameters(self):  # noqa: ANN201
        return list(self._parameters)

    def requires_grad_(self, _enabled: bool) -> "_Model":
        return self

    def to(self, *args, **kwargs) -> "_Model":  # noqa: ANN002, ANN003
        return self

    def train(self) -> None:
        return None

    def enable_gradient_checkpointing(self) -> None:
        return None

    def set_adapter(self, _adapter_name: str) -> None:
        return None

    def enable_adapters(self) -> None:
        return None

    def disable_adapters(self) -> None:
        return None


class _VAE(_Model):
    class _Config:
        scaling_factor = 1.0
        latents_mean = None
        latents_std = None

    def __init__(self) -> None:
        super().__init__(parameter_count=0)
        self.config = self._Config()
        self.dtype = "float32"


class _Optimizer:
    def zero_grad(self) -> None:
        return None

    def step(self) -> None:
        return None


class _Scheduler:
    def __init__(self) -> None:
        self.config = type("Config", (), {"num_train_timesteps": 1000, "prediction_type": "epsilon"})()

    def step(self) -> None:
        return None

    def get_last_lr(self) -> list[float]:
        return [1.0e-4]


class _Loss:
    def detach(self) -> "_Loss":
        return self

    def item(self) -> float:
        return 1.0


class _Accelerator:
    def __init__(self) -> None:
        self.device = "cuda:0"
        self.mixed_precision = "fp16"
        self.sync_gradients = True
        self.is_main_process = True
        self.num_processes = 1

    def prepare(self, *items):  # noqa: ANN002
        return items

    def accumulate(self, _model):  # noqa: ANN001
        return nullcontext()

    def backward(self, _loss) -> None:  # noqa: ANN001
        return None

    def clip_grad_norm_(self, _parameters, _max_grad_norm) -> None:  # noqa: ANN001
        return None

    def log(self, _payload, **_kwargs) -> None:  # noqa: ANN001
        return None

    def save_state(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    def init_trackers(self, *_args, **_kwargs) -> None:  # noqa: ANN002, ANN003
        return None


class _Cuda:
    def __init__(self, *, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return 1 if self._available else 0

    def reset_peak_memory_stats(self) -> None:
        return None

    def max_memory_allocated(self) -> int:
        return 512 * 1024 * 1024


class _Torch:
    def __init__(self, *, cuda_available: bool) -> None:
        self.cuda = _Cuda(available=cuda_available)
        self.float16 = "float16"
        self.bfloat16 = "bfloat16"
        self.float32 = "float32"


def _provenance() -> dict[str, object]:
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
        "diffusers_file": str(REPO_ROOT / "fake_diffusers.py"),
        "accelerate_config_file": str((REPO_ROOT / "configs" / "accelerate" / "single_gpu_fp16.yaml").resolve()),
        "git_commit": "abc123",
        "backend": "uniform",
        "task": "subject_personalization",
        "allocation_manifest": str(MANIFEST_PATH.resolve()),
        "peak_vram_mib": 512.0,
        "timestamp_utc": "2026-04-10T00:00:00Z",
    }


def _install_happy_path(monkeypatch, *, cuda_available: bool = True) -> None:
    fake_unet = _Model(parameter_count=2)
    monkeypatch.setattr(execution, "load_torch", lambda: _Torch(cuda_available=cuda_available))
    monkeypatch.setattr(execution, "create_accelerator", lambda _args: _Accelerator())
    monkeypatch.setattr(
        execution,
        "load_official_sdxl_components",
        lambda **_kwargs: {
            "module": type("Module", (), {"cast_training_params": staticmethod(lambda *_a, **_k: None)})(),
            "tokenizer": object(),
            "tokenizer_2": object(),
            "text_encoder": _Model(parameter_count=0),
            "text_encoder_2": _Model(parameter_count=0),
            "vae": _VAE(),
            "unet": fake_unet,
            "scheduler": object(),
        },
    )
    monkeypatch.setattr(
        execution,
        "build_concrete_adapter_specs",
        lambda _plan, unet: [
            {
                "adapter_name": "uniform_bank",
                "timestep_band": None,
                "cell_ids": [],
                "target_modules": ["down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"],
                "rank": 4,
                "alpha": 4,
                "rank_pattern": {},
                "alpha_pattern": {},
                "noop": False,
            }
        ],
    )

    def fake_apply_adapter_specs(**kwargs) -> None:  # noqa: ANN003
        kwargs["unet"].parameters()[0].requires_grad = True

    monkeypatch.setattr(execution, "apply_adapter_specs", fake_apply_adapter_specs)
    monkeypatch.setattr(execution, "create_train_dataloader", lambda **_kwargs: [{"batch": 1}])
    monkeypatch.setattr(execution, "create_optimizer", lambda **_kwargs: _Optimizer())
    monkeypatch.setattr(execution, "create_lr_scheduler", lambda **_kwargs: _Scheduler())
    monkeypatch.setattr(execution, "compute_step_loss", lambda **_kwargs: (_Loss(), "uniform_bank"))
    monkeypatch.setattr(execution, "build_run_provenance_payload", lambda **_kwargs: _provenance())


def test_real_gpu_hard_fails_when_cuda_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    _install_happy_path(monkeypatch, cuda_available=False)

    with pytest.raises(execution.TrainingExecutionError, match="CUDA"):
        execution.execute_training_run(
            repo_root=REPO_ROOT,
            config=_config(),
            task="subject_personalization",
            backend="uniform",
            allocation_path=MANIFEST_PATH,
            output_dir=tmp_path,
            run_mode="real_gpu",
        )

    for name in execution.REQUIRED_SUCCESS_FILES:
        assert not (tmp_path / name).exists()


def test_success_artifacts_are_not_written_when_checkpoint_save_fails(monkeypatch, tmp_path: Path) -> None:
    _install_happy_path(monkeypatch)
    monkeypatch.setattr(
        execution,
        "save_checkpoint",
        lambda **_kwargs: tmp_path / "checkpoint-1",
    )

    with pytest.raises(execution.TrainingExecutionError, match="Checkpoint"):
        execution.execute_training_run(
            repo_root=REPO_ROOT,
            config=_config(),
            task="subject_personalization",
            backend="uniform",
            allocation_path=MANIFEST_PATH,
            output_dir=tmp_path,
            run_mode="real_gpu",
        )

    for name in execution.REQUIRED_SUCCESS_FILES:
        assert not (tmp_path / name).exists()


@pytest.mark.parametrize("peak_vram_mib", [None, 0.0])
def test_success_artifacts_are_not_written_without_positive_peak_vram(
    monkeypatch,
    tmp_path: Path,
    peak_vram_mib: float | None,
) -> None:
    _install_happy_path(monkeypatch)

    def fake_save_checkpoint(**_kwargs) -> Path:  # noqa: ANN003
        checkpoint_dir = tmp_path / "checkpoint-1"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    monkeypatch.setattr(execution, "save_checkpoint", fake_save_checkpoint)
    monkeypatch.setattr(execution, "_peak_vram_mib_from_torch", lambda _torch: peak_vram_mib)

    with pytest.raises(execution.TrainingExecutionError, match="peak_vram_mib"):
        execution.execute_training_run(
            repo_root=REPO_ROOT,
            config=_config(),
            task="subject_personalization",
            backend="uniform",
            allocation_path=MANIFEST_PATH,
            output_dir=tmp_path,
            run_mode="real_gpu",
        )

    for name in execution.REQUIRED_SUCCESS_FILES:
        assert not (tmp_path / name).exists()


def test_real_gpu_path_cannot_exit_with_planner_status(monkeypatch, tmp_path: Path) -> None:
    _install_happy_path(monkeypatch)

    def fake_save_checkpoint(**_kwargs) -> Path:  # noqa: ANN003
        checkpoint_dir = tmp_path / "checkpoint-1"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    monkeypatch.setattr(execution, "save_checkpoint", fake_save_checkpoint)

    execution.execute_training_run(
        repo_root=REPO_ROOT,
        config=_config(),
        task="subject_personalization",
        backend="uniform",
        allocation_path=MANIFEST_PATH,
        output_dir=tmp_path,
        run_mode="real_gpu",
    )

    summary = json.loads((tmp_path / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] != "planning_only"
