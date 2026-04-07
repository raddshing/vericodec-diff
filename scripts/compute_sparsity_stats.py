from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vericodec_diff.config import OmegaConf
from vericodec_diff.patch_metrics import deep_update, display_path
from vericodec_diff.sparsity_stats import (
    compute_sparsity_stats,
    resolve_sparsity_stats_config,
    write_sparsity_records_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute concentration, Gini, Moran's I, and connected-component sparsity stats."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional repo-relative or absolute YAML config file for sparsity-stat computation.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Target root that receives outputs/metrics/<phase>/ for this run.",
    )
    parser.add_argument(
        "--error-map-root",
        default=None,
        help="Repo-relative or absolute root containing outputs/error_maps/<split>/*.npz.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Repo-relative or absolute root under which outputs/metrics/<phase>/ is created.",
    )
    parser.add_argument(
        "--phase",
        default=None,
        help="Output phase name under outputs/metrics/<phase>/.",
    )
    parser.add_argument(
        "--split-filter",
        default=None,
        help="Optional comma-separated split filter.",
    )
    parser.add_argument(
        "--metric-names",
        default=None,
        help="Optional comma-separated metric list drawn from lpips,one_minus_ssim,hf_wavelet_l1,patch_error.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional NPZ file limit for smoke runs.",
    )
    return parser.parse_args()


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must parse to a mapping")
    return data


def _build_cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {
        "paths": {
            "repo_root": args.repo_root,
        }
    }
    for arg_name, config_key in (
        ("error_map_root", "error_map_root"),
        ("output_root", "output_root"),
        ("phase", "phase"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides["paths"][config_key] = value

    data_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("split_filter", "split_filter"),
        ("metric_names", "metric_names"),
        ("limit", "limit"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            data_overrides[config_key] = value
    if data_overrides:
        overrides["data"] = data_overrides
    return overrides


def main() -> int:
    args = parse_args()
    raw_config: dict[str, object] = {}
    if args.config is not None:
        config_path = Path(args.config).expanduser()
        if not config_path.is_absolute():
            config_path = (REPO_ROOT / config_path).resolve()
        raw_config = _load_yaml_mapping(config_path)

    raw_config = deep_update(raw_config, _build_cli_overrides(args))
    config = resolve_sparsity_stats_config(REPO_ROOT, raw_config)

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = OmegaConf.create(config)
    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(resolved_config, resolved_config_path)

    result = compute_sparsity_stats(config)

    repo_root = Path(config["paths"]["repo_root"])
    records_csv_path = output_dir / "sparsity_stats.csv"
    write_sparsity_records_csv(
        records_csv_path,
        result["records"],
        concentration_percents=config["stats"]["concentration_percents"],
    )

    summary = dict(result["summary"])
    summary["records_csv"] = display_path(records_csv_path, repo_root)
    summary_path = output_dir / "sparsity_stats.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"resolved_config={resolved_config_path}")
    print(f"summary={summary_path}")
    print(f"records_csv={records_csv_path}")
    print(f"output_dir={output_dir}")
    print(f"input_file_count={summary['input_file_count']}")
    print(f"sample_count={summary['sample_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
