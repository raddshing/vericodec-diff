from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.runtime.allocation_manifest import load_allocation_manifest
from rd_lora.substrate.diffusers_sdxl import deep_update, load_yaml_mapping, save_json
from rd_lora.training.adapter_factory import (
    build_backend_plan,
    load_actual_contract_binding,
    write_backend_plan,
)
from vericodec_diff.config import OmegaConf


DEFAULT_CONFIG_PATH = "configs/rdlora_vanilla.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the real RD-LoRA backend contracts and materialize a deterministic training backend plan."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config path.")
    parser.add_argument("--task", choices=("subject_personalization", "style_domain"), required=True)
    parser.add_argument("--backend", choices=("uniform", "layer_only", "timestep_only", "proposed"), required=True)
    parser.add_argument("--allocation", required=True, help="Allocation manifest JSON path.")
    parser.add_argument("--run_mode", choices=("real_gpu",), required=True)
    parser.add_argument("--output_dir", required=True, help="Output directory for the planning artifacts.")
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
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return deep_update(_default_config(), load_yaml_mapping(config_path))


def _routing_mode_summary(plan: dict[str, Any]) -> str:
    routing = dict(plan["routing"])
    if routing["mode"] == "static":
        return f"mode=static adapter={routing['default_adapter_name']}"
    summary = dict(routing["summary"])
    return (
        "mode=timestep_band "
        f"bands={','.join(summary['timestep_bands'])} "
        f"mapped_timestep_total={summary['mapped_timestep_total']}"
    )


def _print_plan_summary(
    *,
    contract_binding: dict[str, Any],
    plan: dict[str, Any],
    allocation_path: Path,
    output_dir: Path,
) -> None:
    probe_rows = ",".join(
        f"{task_name}:{payload['row_count']}"
        for task_name, payload in sorted(contract_binding["probe"].items())
    )
    print(
        "contract_binding "
        f"probe_rows={probe_rows} "
        f"surrogate_model_bytes={contract_binding['surrogate']['model_size_bytes']} "
        f"backends={','.join(contract_binding['allocation']['backends'])}"
    )
    print(
        "backend_plan "
        f"backend={plan['backend']} "
        f"task={plan['task']} "
        f"use_rslora={str(plan['use_rslora']).lower()} "
        f"adapter_banks={len(plan['adapter_banks'])} "
        f"routing={_routing_mode_summary(plan)} "
        f"allocation={allocation_path} "
        f"output_dir={output_dir}"
    )
    for bank in plan["adapter_banks"]:
        parts = [
            f"adapter_name={bank['adapter_name']}",
            f"timestep_band={bank['timestep_band']}",
            f"cell_count={len(bank['cell_ids'])}",
        ]
        if "rank" in bank:
            parts.append(f"rank={bank['rank']}")
        if "alpha" in bank:
            parts.append(f"alpha={bank['alpha']}")
        print("backend_bank " + " ".join(parts))


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allocation_path = Path(args.allocation).expanduser().resolve()
    manifest = load_allocation_manifest(allocation_path)
    if args.backend != manifest["backend"]:
        raise SystemExit(
            f"--backend {args.backend!r} does not match manifest backend {manifest['backend']!r}"
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
                "allocation_manifest": str(allocation_path),
            },
        },
    )
    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(resolved_config), resolved_config_path)

    contract_binding = load_actual_contract_binding()
    plan = build_backend_plan(
        manifest=manifest,
        task=args.task,
        config=resolved_config,
    )

    plan_path = write_backend_plan(plan, output_dir / "adapter_plan.json")
    summary_path = output_dir / "train_summary.json"
    save_json(
        summary_path,
        {
            "status": "planning_only",
            "backend": args.backend,
            "task": args.task,
            "run_mode": args.run_mode,
            "allocation_manifest": str(allocation_path),
            "resolved_config": str(resolved_config_path),
            "backend_plan": str(plan_path),
            "adapter_bank_count": len(plan["adapter_banks"]),
            "routing_mode": plan["routing"]["mode"],
        },
    )

    _print_plan_summary(
        contract_binding=contract_binding,
        plan=plan,
        allocation_path=allocation_path,
        output_dir=output_dir,
    )
    print(f"planning_only=1 summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
