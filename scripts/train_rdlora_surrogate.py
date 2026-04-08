from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.allocator import (
    DEFAULT_ALLOCATOR_CONFIG_PATH,
    resolve_allocator_config,
    train_surrogate_from_config,
)
from rd_lora.substrate.diffusers_sdxl import build_cli_overrides, deep_update, load_yaml_mapping
from vericodec_diff.config import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the deterministic RD-LoRA utility surrogate from probe utility records."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_ALLOCATOR_CONFIG_PATH,
        help="Repo-relative or absolute YAML config for the RD-LoRA Stage C surrogate run.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional repo-root override used for deterministic output paths.",
    )
    parser.add_argument(
        "--probe-run-dir",
        default=None,
        help="Repo-relative or absolute Stage B probe run directory containing utility_records.json.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Repo-relative or absolute root that receives outputs/rd_lora/allocator/<run>/ artifacts.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Single-token run name under outputs/rd_lora/allocator/.",
    )
    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=[],
        help="Override config values with dotted key=value pairs. Repeat as needed.",
    )
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> dict[str, object]:
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    raw_config = load_yaml_mapping(config_path)
    cli_overrides: dict[str, object] = {}
    if args.repo_root is not None:
        cli_overrides.setdefault("paths", {})["repo_root"] = args.repo_root
    if args.probe_run_dir is not None:
        cli_overrides.setdefault("paths", {})["probe_run_dir"] = args.probe_run_dir
    if args.output_root is not None:
        cli_overrides.setdefault("paths", {})["output_root"] = args.output_root
    if args.run_name is not None:
        cli_overrides.setdefault("run", {})["name"] = args.run_name

    raw_config = deep_update(raw_config, cli_overrides)
    raw_config = deep_update(raw_config, build_cli_overrides(args.set_values))
    return resolve_allocator_config(REPO_ROOT, raw_config)


def main() -> int:
    args = parse_args()
    config = _load_config(args)
    run_dir = Path(config["paths"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = run_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(config), resolved_config_path)

    summary = train_surrogate_from_config(config, resolved_config_path=resolved_config_path)
    print(f"resolved_config={resolved_config_path}")
    print(f"model_json={Path(config['paths']['surrogate_model_json'])}")
    print(f"predictions_json={Path(config['paths']['surrogate_predictions_json'])}")
    print(f"predictions_csv={Path(config['paths']['surrogate_predictions_csv'])}")
    print(f"metrics_csv={Path(config['paths']['surrogate_metrics_csv'])}")
    print(f"summary_json={Path(config['paths']['surrogate_summary_json'])}")
    print(f"record_count={summary['record_count']}")
    print(f"cell_count={summary['cell_count']}")
    print(f"preflight_ok={int(summary['preflight']['ok'])}")
    print(f"mae={summary['metrics']['mae']}")
    print(f"rmse={summary['metrics']['rmse']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
