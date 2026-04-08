from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from turbocontext.easyref_runner import (
    build_run_output_dir,
    deep_update,
    display_path,
    resolve_easyref_config,
    run_easyref_baseline,
    save_json,
    write_run_records_csv,
)
from vericodec_diff.config import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the deterministic EasyRef smoke baseline.")
    parser.add_argument(
        "--config",
        default="configs/turbocontext_easyref_baseline.yaml",
        help="Repo-relative or absolute YAML config for the EasyRef baseline wrapper.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Target root that receives outputs/baselines/easyref/ for this run.",
    )
    parser.add_argument(
        "--backend-kind",
        default=None,
        help="Optional backend override: easyref or mock.",
    )
    parser.add_argument(
        "--ref-count",
        type=int,
        default=None,
        help="Optional reference-count override for the smoke baseline run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite any existing generated PNGs for the selected ref-count run.",
    )
    return parser.parse_args()


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must parse to a mapping")
    return data


def _build_cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.repo_root is not None:
        overrides["paths"] = {"repo_root": args.repo_root}
    backend_overrides: dict[str, object] = {}
    if args.backend_kind is not None:
        backend_overrides["kind"] = args.backend_kind
    if backend_overrides:
        overrides["backend"] = backend_overrides

    generation_overrides: dict[str, object] = {}
    if args.ref_count is not None:
        generation_overrides["ref_count"] = args.ref_count
    if args.overwrite:
        generation_overrides["overwrite"] = True
    if generation_overrides:
        overrides["generation"] = generation_overrides
    return overrides


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    raw_config = _load_yaml_mapping(config_path)
    raw_config = deep_update(raw_config, _build_cli_overrides(args))
    config = resolve_easyref_config(REPO_ROOT, raw_config)

    output_dir = build_run_output_dir(config, int(config["generation"]["ref_count"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(config), resolved_config_path)

    result = run_easyref_baseline(config)

    records_csv_path = output_dir / "easyref_run_records.csv"
    write_run_records_csv(records_csv_path, result["records"])

    summary = dict(result["summary"])
    summary["records_csv"] = display_path(records_csv_path, Path(config["paths"]["repo_root"]))
    summary["resolved_config"] = display_path(resolved_config_path, Path(config["paths"]["repo_root"]))
    summary_path = output_dir / "easyref_run_summary.json"
    save_json(summary_path, summary)

    print(f"resolved_config={resolved_config_path}")
    print(f"summary={summary_path}")
    print(f"records_csv={records_csv_path}")
    print(f"output_dir={output_dir}")
    print(f"generated_sample_count={summary['generated_sample_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
