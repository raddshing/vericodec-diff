from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vericodec_diff.config import OmegaConf
from vericodec_diff.reconstruction_records import (
    build_reconstruction_records,
    default_reconstruction_record_config,
    resolve_reconstruction_record_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic reconstruction records from synthetic and real image manifests."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Target root that receives data/ and outputs/ for this record-building run.",
    )
    parser.add_argument(
        "--synthetic-manifest-path",
        default="data/manifests/synthetic_manifest.csv",
        help="Repo-relative or absolute path to the synthetic manifest CSV.",
    )
    parser.add_argument(
        "--real-manifest-path",
        default="data/manifests/real_images_manifest.csv",
        help="Repo-relative or absolute path to the prepared real-image manifest CSV.",
    )
    parser.add_argument(
        "--records-path",
        default="data/manifests/reconstruction_records.csv",
        help="Repo-relative or absolute path for the reconstruction records CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/reconstruction_records",
        help="Repo-relative or absolute directory used for resolved_config.yaml and the summary JSON.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Fraction of each category assigned to the train split.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of each category assigned to the val split.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Fraction of each category assigned to the test split.",
    )
    return parser.parse_args()


def _build_raw_config(args: argparse.Namespace) -> dict[str, object]:
    raw_config = default_reconstruction_record_config()
    raw_config["paths"].update(
        {
            "repo_root": args.repo_root,
            "synthetic_manifest_path": args.synthetic_manifest_path,
            "real_manifest_path": args.real_manifest_path,
            "records_path": args.records_path,
            "output_dir": args.output_dir,
        }
    )
    raw_config["splits"]["fractions"] = {
        "train": args.train_fraction,
        "val": args.val_fraction,
        "test": args.test_fraction,
    }
    return raw_config


def main() -> int:
    args = parse_args()
    config = resolve_reconstruction_record_config(REPO_ROOT, _build_raw_config(args))
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(config), resolved_config_path)

    summary = build_reconstruction_records(config)
    summary_path = output_dir / "build_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"resolved_config={resolved_config_path}")
    print(f"records={config['paths']['records_path']}")
    print(f"summary={summary_path}")
    print(f"record_count={summary['record_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
