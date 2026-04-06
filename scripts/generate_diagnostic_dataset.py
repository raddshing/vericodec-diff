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
from vericodec_diff.synthetic_dataset import (
    LOCKED_SPLIT_COUNTS,
    SYNTHETIC_CATEGORIES,
    default_generation_config,
    generate_dataset,
    parse_count_override,
    parse_csv_items,
    resolve_generation_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic reconstruction-track diagnostic images for VeriCodec-Diff."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Target root that receives data/ and outputs/ for this generation run.",
    )
    parser.add_argument(
        "--raw-root",
        default="data/raw/synthetic",
        help="Repo-relative or absolute directory that receives the PNG images.",
    )
    parser.add_argument(
        "--manifest-path",
        default="data/manifests/synthetic_manifest.csv",
        help="Repo-relative or absolute path for the synthetic manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/diagnostic_dataset",
        help="Repo-relative or absolute directory used for resolved_config.yaml and generation summary.",
    )
    parser.add_argument(
        "--splits",
        default="kill,main,hard",
        help=f"Comma-separated split list drawn from {','.join(LOCKED_SPLIT_COUNTS)}.",
    )
    parser.add_argument(
        "--categories",
        default=",".join(SYNTHETIC_CATEGORIES),
        help=f"Comma-separated category list drawn from {','.join(SYNTHETIC_CATEGORIES)}.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=20260406,
        help="Base seed used to derive deterministic per-image seeds.",
    )
    parser.add_argument(
        "--count-override",
        action="append",
        default=[],
        help="Repeatable split=count override, for example --count-override kill=2.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_config = default_generation_config()
    raw_config["paths"].update(
        {
            "repo_root": args.repo_root,
            "raw_root": args.raw_root,
            "manifest_path": args.manifest_path,
            "output_dir": args.output_dir,
        }
    )
    raw_config["dataset"].update(
        {
            "base_seed": args.base_seed,
            "splits": parse_csv_items(args.splits),
            "categories": parse_csv_items(args.categories),
        }
    )
    for override in args.count_override:
        split, count = parse_count_override(override)
        raw_config["dataset"]["split_counts"][split] = count

    config = resolve_generation_config(REPO_ROOT, raw_config)
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = OmegaConf.create(config)
    OmegaConf.save(resolved_config, output_dir / "resolved_config.yaml")

    summary = generate_dataset(config)
    summary_path = output_dir / "generation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"resolved_config={output_dir / 'resolved_config.yaml'}")
    print(f"manifest={config['paths']['manifest_path']}")
    print(f"summary={summary_path}")
    print(f"image_count={summary['image_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
