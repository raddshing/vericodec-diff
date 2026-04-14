from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vericodec_diff.config import OmegaConf


REQUIRED_COLUMNS = ("prompt_id", "task", "prompt", "seed", "expected_text")


@dataclass(frozen=True)
class ManifestSpec:
    name: str
    task: str
    expected_rows: int
    allowed_seeds: tuple[int, ...]
    prompt_ids: tuple[str, ...]
    require_expected_text: bool


SUBJECT_SPEC = ManifestSpec(
    name="subject_personalization_eval",
    task="subject_personalization",
    expected_rows=32,
    allowed_seeds=(42, 123, 456, 789),
    prompt_ids=tuple(f"subj_{index:02d}" for index in range(1, 9)),
    require_expected_text=False,
)

STYLE_SPEC = ManifestSpec(
    name="style_domain_eval",
    task="style_domain",
    expected_rows=32,
    allowed_seeds=(42, 123),
    prompt_ids=tuple(f"sign_{index:02d}" for index in range(1, 17)),
    require_expected_text=True,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate static RD-LoRA held-out evaluation manifests.")
    parser.add_argument("--subject", required=True, help="Path to subject_personalization_eval.csv.")
    parser.add_argument("--style", required=True, help="Path to style_domain_eval.csv.")
    parser.add_argument(
        "--output_dir",
        default="outputs/validation/rdlora_eval_manifests",
        help="Directory used for resolved_config.yaml and validation_summary.json.",
    )
    return parser.parse_args()


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.loc[:, REQUIRED_COLUMNS].copy()
    normalized["seed"] = pd.to_numeric(normalized["seed"], errors="raise").astype(int)
    for column in ("prompt_id", "task", "prompt", "expected_text"):
        normalized[column] = normalized[column].astype(str)
    return normalized.reset_index(drop=True)


def _sorted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["prompt_id", "seed"], kind="stable").reset_index(drop=True)


def _validate_manifest(path: Path, spec: ManifestSpec) -> dict[str, Any]:
    errors: list[str] = []
    summary: dict[str, Any] = {
        "manifest": spec.name,
        "path": str(path),
        "ok": False,
        "row_count": 0,
        "errors": errors,
    }

    if not path.is_file():
        errors.append(f"{spec.name}: file not found: {path}")
        return summary

    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except Exception as exc:
        errors.append(f"{spec.name}: failed to read CSV: {exc}")
        return summary

    summary["row_count"] = int(len(frame))

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        errors.append(f"{spec.name}: missing required columns: {', '.join(missing_columns)}")
        return summary

    try:
        normalized = _normalize_frame(frame)
    except Exception as exc:
        errors.append(f"{spec.name}: failed to normalize required columns: {exc}")
        return summary

    if len(normalized) != spec.expected_rows:
        errors.append(f"{spec.name}: expected {spec.expected_rows} rows, found {len(normalized)}")

    duplicate_pairs = normalized.duplicated(subset=["prompt_id", "seed"], keep=False)
    if duplicate_pairs.any():
        duplicates = sorted(
            {
                (str(row.prompt_id), int(row.seed))
                for row in normalized.loc[duplicate_pairs, ["prompt_id", "seed"]].itertuples(index=False)
            }
        )
        errors.append(f"{spec.name}: duplicate (prompt_id, seed) pairs: {duplicates}")

    task_values = set(normalized["task"])
    if task_values != {spec.task}:
        errors.append(f"{spec.name}: expected task={spec.task!r}, found {sorted(task_values)}")

    prompt_ids = tuple(sorted(normalized["prompt_id"].unique()))
    if prompt_ids != spec.prompt_ids:
        errors.append(f"{spec.name}: expected prompt_ids {spec.prompt_ids}, found {prompt_ids}")

    blank_prompts = normalized["prompt"].str.strip() == ""
    if blank_prompts.any():
        blank_rows = (blank_prompts[blank_prompts].index + 2).tolist()
        errors.append(f"{spec.name}: blank prompt values at CSV rows {blank_rows}")

    actual_seed_values = tuple(sorted(set(normalized["seed"])))
    invalid_seed_values = tuple(seed for seed in actual_seed_values if seed not in spec.allowed_seeds)
    if invalid_seed_values:
        errors.append(
            f"{spec.name}: invalid seeds {invalid_seed_values}; allowed seeds are {spec.allowed_seeds}"
        )
    if actual_seed_values != spec.allowed_seeds:
        errors.append(f"{spec.name}: expected seed set {spec.allowed_seeds}, found {actual_seed_values}")

    expected_seed_tuple = tuple(spec.allowed_seeds)
    for prompt_id, seed_series in normalized.groupby("prompt_id", sort=True)["seed"]:
        actual_prompt_seeds = tuple(sorted(seed_series.tolist()))
        if actual_prompt_seeds != expected_seed_tuple:
            errors.append(
                f"{spec.name}: prompt_id {prompt_id} expected seeds {expected_seed_tuple}, "
                f"found {actual_prompt_seeds}"
            )

    if spec.require_expected_text:
        empty_expected = normalized["expected_text"].str.strip() == ""
        if empty_expected.any():
            bad_prompt_ids = normalized.loc[empty_expected, "prompt_id"].tolist()
            errors.append(f"{spec.name}: expected_text must be non-empty for all rows; bad prompt_ids={bad_prompt_ids}")
        quoted_text_missing = normalized.apply(
            lambda row: f'"{row["expected_text"]}"' not in row["prompt"],
            axis=1,
        )
        if quoted_text_missing.any():
            bad_prompt_ids = normalized.loc[quoted_text_missing, "prompt_id"].tolist()
            errors.append(
                f"{spec.name}: prompt text must contain the exact quoted expected_text; "
                f"bad prompt_ids={bad_prompt_ids}"
            )
    else:
        non_empty_expected = normalized["expected_text"].str.strip() != ""
        if non_empty_expected.any():
            bad_prompt_ids = normalized.loc[non_empty_expected, "prompt_id"].tolist()
            errors.append(f"{spec.name}: expected_text must be empty for all rows; bad prompt_ids={bad_prompt_ids}")

    if not normalized.equals(_sorted_frame(normalized)):
        errors.append(f"{spec.name}: rows are not sorted by (prompt_id, seed)")

    summary["ok"] = not errors
    return summary


def main() -> int:
    args = parse_args()
    subject_path = _resolve_path(args.subject)
    style_path = _resolve_path(args.style)
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = OmegaConf.create(
        {
            "subject_manifest": str(subject_path),
            "style_manifest": str(style_path),
            "output_dir": str(output_dir),
            "required_columns": list(REQUIRED_COLUMNS),
            "manifest_specs": {
                SUBJECT_SPEC.name: {
                    "task": SUBJECT_SPEC.task,
                    "expected_rows": SUBJECT_SPEC.expected_rows,
                    "allowed_seeds": list(SUBJECT_SPEC.allowed_seeds),
                    "prompt_ids": list(SUBJECT_SPEC.prompt_ids),
                    "require_expected_text": SUBJECT_SPEC.require_expected_text,
                },
                STYLE_SPEC.name: {
                    "task": STYLE_SPEC.task,
                    "expected_rows": STYLE_SPEC.expected_rows,
                    "allowed_seeds": list(STYLE_SPEC.allowed_seeds),
                    "prompt_ids": list(STYLE_SPEC.prompt_ids),
                    "require_expected_text": STYLE_SPEC.require_expected_text,
                },
            },
        }
    )
    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(resolved_config, resolved_config_path)

    summaries = [
        _validate_manifest(subject_path, SUBJECT_SPEC),
        _validate_manifest(style_path, STYLE_SPEC),
    ]
    summary_path = output_dir / "validation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)

    all_errors = [error for summary in summaries for error in summary["errors"]]
    if all_errors:
        for error in all_errors:
            print(error, file=sys.stderr)
        print(f"resolved_config={resolved_config_path}")
        print(f"summary={summary_path}")
        return 1

    for summary in summaries:
        print(f"{summary['manifest']}: rows={summary['row_count']} ok={summary['ok']}")
    print(f"resolved_config={resolved_config_path}")
    print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
