from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.probe import (
    DEFAULT_PROBE_CONFIG_PATH,
    resolve_probe_config,
    run_probe,
    validate_probe_outputs,
)
from rd_lora.substrate.diffusers_sdxl import build_cli_overrides, deep_update, load_yaml_mapping
from vericodec_diff.config import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RD-LoRA week-2 probe instrumentation and write deterministic JSON/CSV artifacts."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_PROBE_CONFIG_PATH,
        help="Repo-relative or absolute YAML config for the RD-LoRA probe run.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional repo-root override used for deterministic output paths.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Single-token run name under outputs/rd_lora/probe/.",
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
    if args.run_name is not None:
        cli_overrides.setdefault("run", {})["name"] = args.run_name
    raw_config = deep_update(raw_config, cli_overrides)
    raw_config = deep_update(raw_config, build_cli_overrides(args.set_values))
    return resolve_probe_config(REPO_ROOT, raw_config)


def main() -> int:
    args = parse_args()
    config = _load_config(args)
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(config), resolved_config_path)

    summary = run_probe(config, resolved_config_path=resolved_config_path)
    validation = validate_probe_outputs(output_dir)

    print(f"resolved_config={resolved_config_path}")
    print(f"summary={output_dir / 'probe_summary.json'}")
    print(f"cell_schema={output_dir / 'cell_schema.json'}")
    print(f"utility_records_json={output_dir / 'utility_records.json'}")
    print(f"utility_records_csv={output_dir / 'utility_records.csv'}")
    print(f"cell_count={summary['cell_count']}")
    print(f"record_count={summary['record_count']}")
    print(f"preflight_ok={int(summary['preflight']['ok'])}")
    print(f"schema_validation_ok={int(validation['ok'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
