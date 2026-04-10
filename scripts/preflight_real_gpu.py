from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.runtime.provenance import collect_runtime_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the real GPU runtime and write a JSON preflight report.")
    parser.add_argument("--repo-root", required=True, help="Repository root used for git commit resolution.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument(
        "--expected-accelerate-config",
        required=True,
        help="Repo-local accelerate config path that must exist.",
    )
    parser.add_argument(
        "--expected-diffusers-substring",
        required=True,
        help="Substring that must appear in the imported diffusers module path.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Require torch.cuda.is_available() == True.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    accelerate_config = Path(args.expected_accelerate_config).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    report = collect_runtime_metadata(
        repo_root=repo_root,
        accelerate_config_file=accelerate_config,
        require_cuda=args.require_cuda,
        expected_diffusers_substring=args.expected_diffusers_substring,
    )
    report = {
        "schema_version": "1.0",
        **report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
