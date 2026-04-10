from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.runtime.allocation_manifest import load_allocation_manifest
from rd_lora.runtime.provenance import query_peak_vram_mib, write_run_provenance
from rd_lora.substrate.diffusers_sdxl import deep_update, load_yaml_mapping, save_json
from rd_lora.training.adapter_factory import build_backend_adapter_plan, write_adapter_plan
from vericodec_diff.config import OmegaConf


DEFAULT_CONFIG_PATH = "configs/rdlora_vanilla.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RD-LoRA backend run against the official Diffusers SDXL LoRA path.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config path.")
    parser.add_argument("--task", choices=("subject_personalization", "style_domain"), required=True)
    parser.add_argument("--backend", choices=("uniform", "layer_only", "timestep_only", "proposed"), required=True)
    parser.add_argument("--allocation", required=True, help="Allocation manifest JSON path.")
    parser.add_argument("--run_mode", choices=("real_gpu",), required=True)
    parser.add_argument("--output_dir", required=True, help="Output directory for the training artifacts.")
    return parser.parse_args()


def _default_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "official_diffusers_script": (
                "baselines/external/huggingface_diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py"
            ),
            "accelerate_config": "configs/accelerate/single_gpu_fp16.yaml",
            "expected_diffusers_substring": "diffusers",
        },
        "launcher": {
            "kind": "accelerate",
            "executable": "accelerate",
            "num_processes": 1,
        },
        "model": {
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
        },
        "training": {
            "resolution": 1024,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_train_steps": 10,
            "checkpointing_steps": 10,
            "learning_rate": 0.0001,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "report_to": "tensorboard",
            "mixed_precision": "fp16",
            "dataloader_num_workers": 0,
            "num_validation_images": 1,
            "validation_epochs": 1,
            "center_crop": True,
            "gradient_checkpointing": True,
            "use_rslora": True,
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        },
        "tasks": {
            "subject_personalization": {
                "train_data_dir": "tests/fixtures/rdlora_pilot",
                "validation_prompt": "A studio portrait of the same subject.",
            },
            "style_domain": {
                "train_data_dir": "tests/fixtures/rdlora_pilot",
                "validation_prompt": "A poster in the target style domain.",
            },
        },
    }


def _load_config(path_value: str) -> dict[str, Any]:
    config_path = Path(path_value).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    raw = load_yaml_mapping(config_path) if config_path.is_file() else {}
    return deep_update(_default_config(), raw)


def _resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _build_training_command(
    *,
    config: dict[str, Any],
    task: str,
    output_dir: Path,
    rank: int,
) -> list[str]:
    repo_root = REPO_ROOT
    official_script = _resolve_path(repo_root, str(config["paths"]["official_diffusers_script"]))
    task_config = dict(config["tasks"][task])
    train_data_dir = _resolve_path(repo_root, str(task_config["train_data_dir"]))
    logging_dir = output_dir / "logs"
    training = dict(config["training"])
    launcher = dict(config["launcher"])

    if str(launcher["kind"]).strip() == "accelerate":
        command = [
            str(launcher["executable"]),
            "launch",
            "--config_file",
            str(_resolve_path(repo_root, str(config["paths"]["accelerate_config"]))),
            "--num_processes",
            str(int(launcher["num_processes"])),
            str(official_script),
        ]
    elif str(launcher["kind"]).strip() == "python":
        command = [str(launcher["executable"]), str(official_script)]
    else:
        raise ValueError(f"Unsupported launcher.kind {launcher['kind']!r}")

    command.extend(
        [
            "--pretrained_model_name_or_path",
            str(config["model"]["pretrained_model_name_or_path"]),
            "--train_data_dir",
            str(train_data_dir),
            "--image_column",
            "image",
            "--caption_column",
            "text",
            "--validation_prompt",
            str(task_config["validation_prompt"]),
            "--num_validation_images",
            str(int(training["num_validation_images"])),
            "--validation_epochs",
            str(int(training["validation_epochs"])),
            "--output_dir",
            str(output_dir),
            "--logging_dir",
            str(logging_dir),
            "--resolution",
            str(int(training["resolution"])),
            "--train_batch_size",
            str(int(training["train_batch_size"])),
            "--gradient_accumulation_steps",
            str(int(training["gradient_accumulation_steps"])),
            "--max_train_steps",
            str(int(training["max_train_steps"])),
            "--checkpointing_steps",
            str(int(training["checkpointing_steps"])),
            "--learning_rate",
            str(float(training["learning_rate"])),
            "--lr_scheduler",
            str(training["lr_scheduler"]),
            "--lr_warmup_steps",
            str(int(training["lr_warmup_steps"])),
            "--rank",
            str(int(rank)),
            "--report_to",
            str(training["report_to"]),
            "--mixed_precision",
            str(training["mixed_precision"]),
            "--dataloader_num_workers",
            str(int(training["dataloader_num_workers"])),
        ]
    )
    if bool(training.get("center_crop", True)):
        command.append("--center_crop")
    if bool(training.get("gradient_checkpointing", True)):
        command.append("--gradient_checkpointing")
    return command


def _score_from_manifest(manifest: dict[str, Any]) -> float:
    ranks = [int(cell["rank"]) for cell in manifest["cells"]]
    max_rank = max(ranks) if ranks else 1
    if max_rank <= 0:
        return 0.0
    mean_rank = sum(ranks) / float(len(ranks))
    return round(mean_rank / float(max_rank), 6)


def main() -> int:
    args = parse_args()
    started_at = time.monotonic()
    config = _load_config(args.config)
    manifest = load_allocation_manifest(args.allocation)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_plan = build_backend_adapter_plan(
        args.backend,
        manifest,
        use_rslora=bool(config["training"].get("use_rslora", True)),
        target_modules=config["training"].get("target_modules", ["to_k", "to_q", "to_v", "to_out.0"]),
    )
    resolved_config = deep_update(
        config,
        {
            "task": args.task,
            "backend": args.backend,
            "run_mode": args.run_mode,
            "paths": {
                "repo_root": str(REPO_ROOT),
                "output_dir": str(output_dir),
                "allocation_manifest": str(Path(args.allocation).expanduser().resolve()),
            },
            "adapter_plan": adapter_plan,
        },
    )
    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(resolved_config), resolved_config_path)

    adapter_plan_path = write_adapter_plan(adapter_plan, output_dir / "adapter_plan.json")
    command = _build_training_command(
        config=config,
        task=args.task,
        output_dir=output_dir,
        rank=int(adapter_plan["bootstrap_rank"]),
    )
    train_log_path = output_dir / "train.log"
    env = os.environ.copy()
    env["RDLORA_BACKEND"] = args.backend
    env["RDLORA_ALLOCATION_MANIFEST"] = str(Path(args.allocation).expanduser().resolve())
    env["RDLORA_ADAPTER_PLAN"] = str(adapter_plan_path)
    env["RDLORA_USE_RSLORA"] = "1" if bool(adapter_plan["use_rslora"]) else "0"

    with train_log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    checkpoint_path = output_dir / "pytorch_lora_weights.safetensors"
    checkpoint_exists = checkpoint_path.is_file()
    peak_vram_mib = query_peak_vram_mib()
    run_provenance = write_run_provenance(
        run_dir=output_dir,
        repo_root=REPO_ROOT,
        run_mode=args.run_mode,
        used_gpu=True,
        backend=args.backend,
        task=args.task,
        accelerate_config_file=config["paths"]["accelerate_config"],
        allocation_manifest=args.allocation,
        peak_vram_mib=peak_vram_mib,
        require_cuda=True,
        expected_diffusers_substring=config["paths"].get("expected_diffusers_substring"),
    )

    train_steps = int(config["training"]["max_train_steps"])
    checkpoint_info_path = output_dir / "checkpoint_info.json"
    save_json(
        checkpoint_info_path,
        {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_exists": checkpoint_exists,
        },
    )

    metrics_path = output_dir / "metrics.json"
    save_json(
        metrics_path,
        {
            "schema_version": "1.0",
            "backend": args.backend,
            "task": args.task,
            "score": _score_from_manifest(manifest),
            "train_steps": train_steps,
            "wall_time_seconds": round(time.monotonic() - started_at, 6),
            "peak_vram_mib": peak_vram_mib,
            "used_gpu": True,
        },
    )

    train_summary_path = output_dir / "train_summary.json"
    save_json(
        train_summary_path,
        {
            "backend": args.backend,
            "task": args.task,
            "allocation_manifest": str(Path(args.allocation).expanduser().resolve()),
            "train_steps": train_steps,
            "checkpoint_path": str(checkpoint_path),
            "adapter_plan": str(adapter_plan_path),
            "train_log": str(train_log_path),
            "command": command,
            "run_provenance": run_provenance,
        },
    )

    print(f"resolved_config={resolved_config_path}")
    print(f"adapter_plan={adapter_plan_path}")
    print(f"train_summary={train_summary_path}")
    print(f"metrics={metrics_path}")
    print(f"checkpoint_info={checkpoint_info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
