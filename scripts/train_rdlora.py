from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.substrate.diffusers_sdxl import deep_update, load_yaml_mapping
from rd_lora.training.execution import execute_training_run
from vericodec_diff.config import OmegaConf


DEFAULT_CONFIG_PATH = "configs/rdlora_gate.yaml"


def _parse_forced_timestep_band_sequence(value: str | None) -> list[str] | None:
    if value is None:
        return None

    bands = [segment.strip() for segment in str(value).split(",")]
    if any(not band for band in bands):
        raise argparse.ArgumentTypeError(
            "--forced_timestep_band_sequence must be a comma-separated list of non-empty band names"
        )
    return bands or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute real SDXL DreamBooth LoRA training for the RD-LoRA backends."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config path.")
    parser.add_argument("--task", choices=("subject_personalization", "style_domain"), required=True)
    parser.add_argument("--backend", choices=("uniform", "layer_only", "timestep_only", "proposed"), required=True)
    parser.add_argument("--allocation", required=True, help="Allocation manifest JSON path.")
    parser.add_argument("--run_mode", choices=("real_gpu",), required=True)
    parser.add_argument("--output_dir", required=True, help="Output directory for the run artifacts.")
    parser.add_argument(
        "--forced_timestep_band_sequence",
        type=_parse_forced_timestep_band_sequence,
        default=None,
        help="Optional comma-separated timestep-band sequence for deterministic smoke routing.",
    )
    return parser.parse_args()


def _default_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "official_diffusers_script": (
                "baselines/external/huggingface_diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py"
            ),
            "accelerate_config": "configs/accelerate/single_gpu_fp16.yaml",
            "expected_diffusers_substring": "huggingface_diffusers",
        },
        "model": {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
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
            "random_flip": False,
            "gradient_checkpointing": True,
            "scale_lr": False,
            "allow_tf32": False,
            "use_8bit_adam": False,
            "enable_xformers_memory_efficient_attention": False,
            "optimizer": "AdamW",
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_weight_decay": 1.0e-4,
            "adam_weight_decay_text_encoder": 1.0e-3,
            "adam_epsilon": 1.0e-8,
            "max_grad_norm": 1.0,
            "lora_dropout": 0.0,
            "repeats": 1,
            "seed": 20260410,
            "use_rslora": True,
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        },
        "tasks": {
            "subject_personalization": {
                "train_data_dir": "tests/fixtures/rdlora_pilot",
                "instance_prompt": "a studio portrait of the subject",
                "validation_prompt": "A studio portrait of the same subject.",
            },
            "style_domain": {
                "train_data_dir": "tests/fixtures/rdlora_pilot",
                "instance_prompt": "an image rendered in the target style domain",
                "validation_prompt": "A poster in the target style domain.",
            },
        },
    }


def _load_config(path_value: str) -> dict[str, Any]:
    config_path = Path(path_value).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return deep_update(_default_config(), load_yaml_mapping(config_path))


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = deep_update(
        config,
        {
            "task": args.task,
            "backend": args.backend,
            "run_mode": args.run_mode,
            "training": {
                "forced_timestep_band_sequence": args.forced_timestep_band_sequence,
            },
            "paths": {
                "repo_root": str(REPO_ROOT),
                "output_dir": str(output_dir),
                "allocation_manifest": str(Path(args.allocation).expanduser().resolve()),
            },
        },
    )
    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(resolved_config), resolved_config_path)

    result = execute_training_run(
        repo_root=REPO_ROOT,
        config=resolved_config,
        task=args.task,
        backend=args.backend,
        allocation_path=args.allocation,
        output_dir=output_dir,
        run_mode=args.run_mode,
        forced_timestep_band_sequence=args.forced_timestep_band_sequence,
    )
    print(f"resolved_config={resolved_config_path}")
    print(f"backend={args.backend}")
    print(f"task={args.task}")
    print(f"train_steps={result['train_summary']['train_steps']}")
    print(f"checkpoint={result['checkpoint_info']['checkpoint_path']}")
    print(f"peak_vram_mib={result['run_provenance']['peak_vram_mib']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
