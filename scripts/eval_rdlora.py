from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.stage_d import (
    DEFAULT_STAGE_D_CONFIG_PATH,
    resolve_stage_d_config,
    write_stage_d_evaluation,
)
from rd_lora.substrate.diffusers_sdxl import build_cli_overrides, deep_update, load_yaml_mapping
from vericodec_diff.config import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage D allocations and compute the RD-LoRA week-2 gate metrics."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_STAGE_D_CONFIG_PATH,
        help="Repo-relative or absolute YAML config for the Stage D evaluation harness.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Optional repo-root override used for deterministic output paths.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Single-token run name under outputs/rd_lora/stage_d/.",
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
    return resolve_stage_d_config(REPO_ROOT, raw_config)


def main() -> int:
    args = parse_args()
    config = _load_config(args)
    evaluation_dir = Path(config["paths"]["evaluation_dir"])
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = evaluation_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(config), resolved_config_path)

    result = write_stage_d_evaluation(config)
    payload = result["payload"]

    print(f"resolved_config={resolved_config_path}")
    print(f"evaluation_summary={result['evaluation_summary_path']}")
    print(f"baseline_table={result['baseline_table_path']}")
    print(f"decision={payload['decision']}")
    print(f"top_20_mass_ratio={payload['measured']['top_20_mass_ratio']}")
    print(f"held_out_spearman_rho={payload['measured']['held_out_spearman_rho']}")
    print(f"top_5_precision={payload['measured']['top_5_precision']}")
    print(f"proposed_relative_vs_uniform={payload['measured']['proposed_relative_vs_uniform']}")
    print(f"probe_allocation_overhead_ratio={payload['measured']['probe_allocation_overhead_ratio']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
