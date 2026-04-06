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
from vericodec_diff.prompt_manifests import MANIFEST_SPECS, validate_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate canonical VeriCodec-Diff prompt manifests.")
    parser.add_argument(
        "--manifests-dir",
        default="data/manifests",
        help="Directory containing the canonical prompt manifest CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/validation/prompt_manifests",
        help="Directory used for resolved_config.yaml and validation summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifests_dir = (REPO_ROOT / args.manifests_dir).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OmegaConf.create(
        {
            "manifests_dir": str(manifests_dir),
            "output_dir": str(output_dir),
            "manifest_names": list(MANIFEST_SPECS),
        }
    )
    OmegaConf.save(config, output_dir / "resolved_config.yaml")

    manifest_paths = [manifests_dir / name for name in MANIFEST_SPECS]
    summaries = validate_manifests(manifest_paths)
    summary_path = output_dir / "validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)

    for summary in summaries:
        print(
            f"{summary['manifest']}: total={summary['total']} "
            f"categories={json.dumps(summary['category_counts'], sort_keys=True)}"
        )
    print(f"resolved_config={output_dir / 'resolved_config.yaml'}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
