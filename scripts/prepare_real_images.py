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
from vericodec_diff.real_images import default_real_image_config, prepare_real_images, resolve_real_image_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic 1024x1024 real-image inputs for the VeriCodec-Diff kill test."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Target root that receives data/ and outputs/ for this preparation run.",
    )
    parser.add_argument(
        "--prepared-root",
        default="data/raw/real_prepared",
        help="Repo-relative or absolute directory that receives normalized PNG images.",
    )
    parser.add_argument(
        "--manifest-path",
        default="data/manifests/real_images_manifest.csv",
        help="Repo-relative or absolute path for the prepared real-image manifest CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/real_image_prep",
        help="Repo-relative or absolute directory used for resolved_config.yaml and preparation summary.",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--ffhq-root",
        help="Local FFHQ root. Images are discovered recursively under this directory.",
    )
    source_group.add_argument(
        "--openimages-root",
        help="Local OpenImages root with class subdirectories.",
    )
    source_group.add_argument(
        "--hf-dataset",
        help="Hugging Face dataset name or local dataset builder, for example laion/laion-art or imagefolder.",
    )
    parser.add_argument(
        "--openimages-class",
        action="append",
        default=[],
        help="Optional class subdirectory under --openimages-root. Repeat to combine multiple classes.",
    )
    parser.add_argument(
        "--hf-split",
        default="train",
        help="Split name to load when using --hf-dataset.",
    )
    parser.add_argument(
        "--hf-config-name",
        help="Optional Hugging Face config name.",
    )
    parser.add_argument(
        "--hf-data-dir",
        help="Optional local data_dir passed through to datasets.load_dataset.",
    )
    parser.add_argument(
        "--hf-image-column",
        help="Optional Hugging Face image column override.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        help="Optional repo-relative or absolute cache_dir passed through to datasets.load_dataset().",
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Output category label used for the prepared files and manifest rows.",
    )
    parser.add_argument(
        "--split",
        help="Manifest split label for the ingested rows. Defaults to the HF split or 'unspecified' for local sources.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional deterministic cap on the number of selected source images.",
    )
    parser.add_argument(
        "--source-name",
        help="Optional source_name override written into the manifest.",
    )
    parser.add_argument(
        "--source-url",
        help="Optional source_url override written into the manifest.",
    )
    parser.add_argument(
        "--license",
        default="unknown",
        help="License string recorded for each prepared image.",
    )
    return parser.parse_args()


def _build_raw_config(args: argparse.Namespace) -> dict[str, object]:
    raw_config = default_real_image_config()
    raw_config["paths"].update(
        {
            "repo_root": args.repo_root,
            "prepared_root": args.prepared_root,
            "manifest_path": args.manifest_path,
            "output_dir": args.output_dir,
        }
    )
    raw_config["selection"].update(
        {
            "category": args.category,
            "split": args.split,
            "limit": args.limit,
            "source_name": args.source_name,
            "source_url": args.source_url,
            "license": args.license,
        }
    )

    if args.ffhq_root:
        raw_config["source"].update({"kind": "ffhq", "root": args.ffhq_root})
    elif args.openimages_root:
        raw_config["source"].update(
            {
                "kind": "openimages",
                "root": args.openimages_root,
                "openimages_classes": list(args.openimages_class),
            }
        )
    else:
        raw_config["source"].update(
            {
                "kind": "huggingface",
                "hf_dataset": args.hf_dataset,
                "hf_split": args.hf_split,
                "hf_config_name": args.hf_config_name,
                "hf_data_dir": args.hf_data_dir,
                "hf_image_column": args.hf_image_column,
                "hf_cache_dir": args.hf_cache_dir,
            }
        )
    return raw_config


def main() -> int:
    args = parse_args()
    config = resolve_real_image_config(REPO_ROOT, _build_raw_config(args))
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(config), resolved_config_path)

    summary = prepare_real_images(config)
    summary_path = output_dir / "preparation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"resolved_config={resolved_config_path}")
    print(f"manifest={config['paths']['manifest_path']}")
    print(f"summary={summary_path}")
    print(f"prepared_count={summary['prepared_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
