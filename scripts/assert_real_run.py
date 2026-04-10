from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.runtime.provenance import (
    FORBIDDEN_SOURCE_TOKENS,
    assert_real_gpu_provenance,
    iter_recorded_source_paths,
    load_run_provenance,
    path_contains_forbidden_token,
)
from rd_lora.training.execution import validate_training_output_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assert that a run directory contains a real GPU RD-LoRA run.")
    parser.add_argument("--run_dir", required=True, help="Run directory that contains run_provenance.json.")
    parser.add_argument("--require-gpu", action="store_true", help="Require GPU usage and positive peak VRAM.")
    parser.add_argument("--forbid-mock", action="store_true", help="Reject mock/smoke source paths.")
    parser.add_argument(
        "--require-file",
        action="append",
        default=[],
        help="Repeatable relative path that must exist under run_dir.",
    )
    parser.add_argument(
        "--require-training-success",
        action="store_true",
        help="Require the full training success artifact contract.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    provenance = load_run_provenance(run_dir / "run_provenance.json")
    assert_real_gpu_provenance(
        provenance,
        require_gpu=args.require_gpu,
        forbid_mock=args.forbid_mock,
    )
    for relative_path in args.require_file:
        required_path = run_dir / relative_path
        if not required_path.exists():
            raise FileNotFoundError(f"Required file missing: {required_path}")
    if args.require_training_success:
        validate_training_output_artifacts(run_dir)
    if args.forbid_mock:
        if path_contains_forbidden_token(run_dir, FORBIDDEN_SOURCE_TOKENS):
            raise RuntimeError(f"Forbidden mock/smoke token found in run_dir: {run_dir}")
        for source_path in iter_recorded_source_paths(provenance):
            if path_contains_forbidden_token(source_path, FORBIDDEN_SOURCE_TOKENS):
                raise RuntimeError(f"Forbidden mock/smoke token found in source path: {source_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
