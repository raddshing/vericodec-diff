from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys


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
            "pretrained_vae_model_name_or_path": None,
            "revision": None,
            "variant": None,
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


class FakeParam:
    def __init__(self, *, requires_grad: bool = False) -> None:
        self.requires_grad = requires_grad


class FakeModel:
    def __init__(self, *, parameter_count: int = 2) -> None:
        self._parameters = [FakeParam(requires_grad=False) for _ in range(parameter_count)]
        self.peft_config: dict[str, object] = {}

    def parameters(self):  # noqa: ANN201
        return list(self._parameters)

    def named_modules(self):  # noqa: ANN201
        return []

    def requires_grad_(self, enabled: bool) -> "FakeModel":
        return self

    def to(self, *args, **kwargs) -> "FakeModel":  # noqa: ANN002, ANN003
        return self

    def train(self) -> None:
        return None

    def add_adapter(self, adapter_config, adapter_name: str = "default") -> None:  # noqa: ANN001
        self.peft_config[adapter_name] = adapter_config

    def set_adapter(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name

    def enable_adapters(self) -> None:
        self.adapters_enabled = True

    def disable_adapters(self) -> None:
        self.adapters_enabled = False

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True


class FakeVAE(FakeModel):
    class _Config:
        scaling_factor = 1.0
        latents_mean = None
        latents_std = None

    def __init__(self) -> None:
        super().__init__(parameter_count=0)
        self.config = self._Config()
        self.dtype = "float32"


class FakeScheduler:
    def __init__(self) -> None:
        self.step_calls = 0
        self.config = type("Config", (), {"num_train_timesteps": 1000, "prediction_type": "epsilon"})()

    def step(self) -> None:
        self.step_calls += 1

    def get_last_lr(self) -> list[float]:
        return [1.0e-4]


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


class FakeAccelerator:
    def __init__(self) -> None:
        self.device = "cuda:0"
        self.mixed_precision = "fp16"
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


def _provenance(checkpoint_dir: Path) -> dict[str, object]:
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
        "diffusers_file": str(checkpoint_dir / "fake_diffusers.py"),
        "accelerate_config_file": str((REPO_ROOT / "configs" / "accelerate" / "single_gpu_fp16.yaml").resolve()),
        "git_commit": "abc123",
        "backend": "uniform",
        "task": "subject_personalization",
        "allocation_manifest": str(MANIFEST_PATH.resolve()),
        "peak_vram_mib": 512.0,
        "timestamp_utc": "2026-04-10T00:00:00Z",
    }


def test_real_execution_path_loads_components_and_runs_backward_step_checkpoint(monkeypatch, tmp_path: Path) -> None:
    fake_torch = FakeTorch()
    fake_unet = FakeModel()
    fake_components = {
        "module": type("Module", (), {"cast_training_params": staticmethod(lambda *_a, **_k: None)})(),
        "tokenizer": object(),
        "tokenizer_2": object(),
        "text_encoder": FakeModel(parameter_count=0),
        "text_encoder_2": FakeModel(parameter_count=0),
        "vae": FakeVAE(),
        "unet": fake_unet,
        "scheduler": object(),
    }
    fake_accelerator = FakeAccelerator()
    fake_scheduler = FakeScheduler()
    events: list[str] = []

    monkeypatch.setattr(execution, "load_torch", lambda: fake_torch)
    monkeypatch.setattr(execution, "create_accelerator", lambda _args: fake_accelerator)
    monkeypatch.setattr(execution, "load_official_sdxl_components", lambda **_kwargs: fake_components)
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
        kwargs["unet"].parameters()[1].requires_grad = True

    monkeypatch.setattr(execution, "apply_adapter_specs", fake_apply_adapter_specs)
    monkeypatch.setattr(execution, "create_train_dataloader", lambda **_kwargs: [{"batch": 1}])

    optimizer_holder: dict[str, FakeOptimizer] = {}

    def fake_create_optimizer(*, trainable_parameters, **_kwargs):  # noqa: ANN003
        optimizer = FakeOptimizer(trainable_parameters)
        optimizer_holder["optimizer"] = optimizer
        return optimizer

    monkeypatch.setattr(execution, "create_optimizer", fake_create_optimizer)
    monkeypatch.setattr(execution, "create_lr_scheduler", lambda **_kwargs: fake_scheduler)
    monkeypatch.setattr(
        execution,
        "compute_step_loss",
        lambda **_kwargs: (FakeLoss(1.25), "uniform_bank"),
    )

    def fake_build_run_provenance_payload(**_kwargs):  # noqa: ANN003
        return _provenance(tmp_path)

    monkeypatch.setattr(execution, "build_run_provenance_payload", fake_build_run_provenance_payload)

    def fake_save_checkpoint(*, accelerator, output_dir, global_step):  # noqa: ANN001
        checkpoint_dir = output_dir / f"checkpoint-{global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        events.append("checkpoint")
        return checkpoint_dir

    monkeypatch.setattr(execution, "save_checkpoint", fake_save_checkpoint)
    original_write_success_artifacts = execution.write_success_artifacts

    def wrapped_write_success_artifacts(**kwargs) -> None:  # noqa: ANN003
        assert Path(kwargs["checkpoint_info"]["checkpoint_path"]).exists()
        events.append("artifacts")
        original_write_success_artifacts(**kwargs)

    monkeypatch.setattr(execution, "write_success_artifacts", wrapped_write_success_artifacts)

    result = execution.execute_training_run(
        repo_root=REPO_ROOT,
        config=_config(),
        task="subject_personalization",
        backend="uniform",
        allocation_path=MANIFEST_PATH,
        output_dir=tmp_path,
        run_mode="real_gpu",
    )

    assert fake_torch.cuda.reset_called is True
    assert fake_accelerator.backward_calls and fake_accelerator.backward_calls[0].item() == 1.25
    assert optimizer_holder["optimizer"].zero_grad_calls == 1
    assert optimizer_holder["optimizer"].step_calls == 1
    assert fake_scheduler.step_calls == 1
    assert [parameter for parameter in optimizer_holder["optimizer"].parameters if parameter.requires_grad] == [
        fake_unet.parameters()[1]
    ]
    assert events == ["checkpoint", "artifacts"]
    assert result["checkpoint_info"]["checkpoint_exists"] is True
    assert (tmp_path / "run_provenance.json").is_file()
    assert (tmp_path / "train_summary.json").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "checkpoint_info.json").is_file()
