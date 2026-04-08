from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.substrate.diffusers_sdxl import (
    build_cli_overrides,
    build_diffusers_sdxl_lora_command,
    deep_update,
    display_path,
    load_rdlora_tasks,
    load_yaml_mapping,
    prepare_pilot_imagefolder,
    resolve_vanilla_lora_config,
    save_json,
    validate_launch_plan,
    write_shell_command,
)
from vericodec_diff.config import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only smoke validation for the RD-LoRA SDXL command wrapper."
    )
    parser.add_argument(
        "--config",
        default="configs/rdlora_vanilla.yaml",
        help="Repo-relative or absolute YAML config for the RD-LoRA SDXL wrapper.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional repo-root override used for deterministic output paths.",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="Task id from configs/rdlora_tasks.yaml. Defaults to smoke.task_id from the wrapper config.",
    )
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        help="Override config values with dotted key=value pairs. Repeat as needed.",
    )
    return parser.parse_args()


def _load_wrapper_config(args: argparse.Namespace) -> dict[str, object]:
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    raw_config = load_yaml_mapping(config_path)
    cli_overrides: dict[str, object] = {}
    if args.repo_root is not None:
        cli_overrides["paths"] = {"repo_root": args.repo_root}
    raw_config = deep_update(raw_config, cli_overrides)
    raw_config = deep_update(raw_config, build_cli_overrides(args.set_values))
    return resolve_vanilla_lora_config(REPO_ROOT, raw_config)


def main() -> int:
    args = parse_args()
    config = _load_wrapper_config(args)
    task_id = args.task_id or str(config["smoke"]["task_id"])

    smoke_root = Path(config["paths"]["smoke_root"]) / task_id / str(config["smoke"]["run_name"])
    smoke_root.mkdir(parents=True, exist_ok=True)
    resolved_config_path = smoke_root / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(config), resolved_config_path)

    tasks_bundle = load_rdlora_tasks(config["paths"]["tasks_config"])
    prepared = prepare_pilot_imagefolder(config, tasks_bundle, task_id)
    plan = build_diffusers_sdxl_lora_command(
        config,
        task_id=task_id,
        run_name=str(config["smoke"]["run_name"]),
        smoke=True,
    )
    repo_root = Path(config["paths"]["repo_root"])
    validation = validate_launch_plan(plan, repo_root=repo_root)

    launch_script_path = smoke_root / "manual_gpu_command.sh"
    write_shell_command(launch_script_path, plan["command"])
    summary_payload = {
        "task_id": task_id,
        "prepared_image_count": prepared["image_count"],
        "wrapped_official_script": validation["wrapped_official_script"],
        "official_script": validation["official_script"],
        "train_data_dir": validation["train_data_dir"],
        "output_dir": validation["output_dir"],
        "resolved_config": display_path(resolved_config_path, repo_root),
        "manual_gpu_command_script": display_path(launch_script_path, repo_root),
        "manual_gpu_command": plan["manual_command"],
        "accelerate_found": validation["accelerate_found"],
    }
    summary_path = smoke_root / "smoke_summary.json"
    save_json(summary_path, summary_payload)

    print(f"resolved_config={resolved_config_path}")
    print(f"summary={summary_path}")
    print(f"manual_gpu_command={plan['manual_command']}")
    print(f"wrapped_official_script={int(validation['wrapped_official_script'])}")
    print(f"prepared_image_count={prepared['image_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
