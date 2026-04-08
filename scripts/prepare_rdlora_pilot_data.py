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
    deep_update,
    display_path,
    load_rdlora_tasks,
    load_yaml_mapping,
    prepare_pilot_imagefolder,
    resolve_task_layout,
    resolve_vanilla_lora_config,
    save_json,
)
from vericodec_diff.config import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a deterministic pilot imagefolder dataset for the RD-LoRA SDXL harness."
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
    tasks_bundle = load_rdlora_tasks(config["paths"]["tasks_config"])
    task_id = args.task_id or str(config["smoke"]["task_id"])
    layout = resolve_task_layout(config, tasks_bundle, task_id)
    task_root = Path(layout["task_root"])
    task_root.mkdir(parents=True, exist_ok=True)

    resolved_config_path = task_root / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(config), resolved_config_path)

    summary = prepare_pilot_imagefolder(config, tasks_bundle, task_id)
    repo_root = Path(config["paths"]["repo_root"])
    summary_payload = {
        "task_id": task_id,
        "concept_name": summary["concept_name"],
        "image_count": summary["image_count"],
        "train_data_dir": display_path(Path(summary["train_data_dir"]), repo_root),
        "manifest_path": display_path(Path(summary["manifest_path"]), repo_root),
        "metadata_path": display_path(Path(summary["metadata_path"]), repo_root),
        "resolved_config": display_path(resolved_config_path, repo_root),
    }
    summary_path = task_root / "pilot_data_summary.json"
    save_json(summary_path, summary_payload)

    print(f"resolved_config={resolved_config_path}")
    print(f"summary={summary_path}")
    print(f"manifest={summary['manifest_path']}")
    print(f"metadata={summary['metadata_path']}")
    print(f"train_data_dir={summary['train_data_dir']}")
    print(f"image_count={summary['image_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
