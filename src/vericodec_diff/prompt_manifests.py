from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = (
    "prompt_id",
    "category",
    "stress_type",
    "prompt",
    "negative_prompt",
    "seed",
    "split",
)

ALLOWED_CATEGORIES = (
    "text-heavy",
    "thin-boundary",
    "repeated-pattern",
    "faces",
    "mixed scenes",
)

MANIFEST_SPECS = {
    "prompt_manifest_kill.csv": {
        "split": "kill",
        "total": 240,
        "category_counts": {
            "text-heavy": 48,
            "thin-boundary": 48,
            "repeated-pattern": 48,
            "faces": 48,
            "mixed scenes": 48,
        },
    },
    "prompt_manifest_main.csv": {
        "split": "main",
        "total": 500,
        "category_counts": {
            "text-heavy": 100,
            "thin-boundary": 100,
            "repeated-pattern": 100,
            "faces": 100,
            "mixed scenes": 100,
        },
    },
    "prompt_manifest_hard.csv": {
        "split": "hard",
        "total": 150,
        "category_counts": {
            "text-heavy": 30,
            "thin-boundary": 30,
            "repeated-pattern": 30,
            "faces": 30,
            "mixed scenes": 30,
        },
    },
}

_PROMPT_ID_RE = re.compile(
    r"^(kill|main|hard)_(text_heavy|thin_boundary|repeated_pattern|faces|mixed_scenes)_[0-9]{3}$"
)
_TEXT_HEAVY_QUOTE_RE = re.compile(r'"[^"\n]+"')


class ManifestValidationError(ValueError):
    """Raised when a prompt manifest violates the repo-local schema."""


def _category_to_id_token(category: str) -> str:
    return category.replace("-", "_").replace(" ", "_")


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != REQUIRED_COLUMNS:
            raise ManifestValidationError(
                f"{path.name}: expected columns {REQUIRED_COLUMNS}, found {fieldnames}"
            )
        return list(reader)


def _validate_row(
    row: dict[str, str],
    *,
    path: Path,
    row_number: int,
    expected_split: str,
) -> None:
    for column in REQUIRED_COLUMNS:
        value = row.get(column, "")
        if value is None or not str(value).strip():
            raise ManifestValidationError(
                f"{path.name}: row {row_number} has empty value for {column}"
            )

    if row["category"] not in ALLOWED_CATEGORIES:
        raise ManifestValidationError(
            f"{path.name}: row {row_number} has unsupported category {row['category']!r}"
        )

    if row["split"] != expected_split:
        raise ManifestValidationError(
            f"{path.name}: row {row_number} has split {row['split']!r}, expected {expected_split!r}"
        )

    if not _PROMPT_ID_RE.match(row["prompt_id"]):
        raise ManifestValidationError(
            f"{path.name}: row {row_number} has invalid prompt_id {row['prompt_id']!r}"
        )

    expected_prefix = f"{expected_split}_{_category_to_id_token(row['category'])}_"
    if not row["prompt_id"].startswith(expected_prefix):
        raise ManifestValidationError(
            f"{path.name}: row {row_number} prompt_id {row['prompt_id']!r} does not match split/category"
        )

    try:
        seed = int(row["seed"])
    except ValueError as exc:
        raise ManifestValidationError(
            f"{path.name}: row {row_number} has non-integer seed {row['seed']!r}"
        ) from exc

    if seed <= 0:
        raise ManifestValidationError(
            f"{path.name}: row {row_number} must use a positive seed, found {seed}"
        )

    if row["category"] == "text-heavy" and not _TEXT_HEAVY_QUOTE_RE.search(row["prompt"]):
        raise ManifestValidationError(
            f"{path.name}: row {row_number} text-heavy prompt must contain an explicit quoted string"
        )


def validate_manifest(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path)
    if manifest_path.name not in MANIFEST_SPECS:
        raise ManifestValidationError(
            f"{manifest_path.name}: no manifest spec registered for this file"
        )

    spec = MANIFEST_SPECS[manifest_path.name]
    rows = _load_rows(manifest_path)
    if len(rows) != spec["total"]:
        raise ManifestValidationError(
            f"{manifest_path.name}: expected {spec['total']} rows, found {len(rows)}"
        )

    prompt_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    stress_counts: Counter[str] = Counter()

    for row_number, row in enumerate(rows, start=2):
        _validate_row(row, path=manifest_path, row_number=row_number, expected_split=spec["split"])

        if row["prompt_id"] in prompt_ids:
            raise ManifestValidationError(
                f"{manifest_path.name}: duplicate prompt_id {row['prompt_id']!r}"
            )
        prompt_ids.add(row["prompt_id"])

        category_counts[row["category"]] += 1
        stress_counts[row["stress_type"]] += 1

    expected_counts = spec["category_counts"]
    if dict(category_counts) != expected_counts:
        raise ManifestValidationError(
            f"{manifest_path.name}: expected category counts {expected_counts}, found {dict(category_counts)}"
        )

    return {
        "manifest": manifest_path.name,
        "split": spec["split"],
        "total": len(rows),
        "category_counts": dict(category_counts),
        "stress_type_count": len(stress_counts),
    }


def validate_manifests(paths: Iterable[str | Path]) -> list[dict[str, object]]:
    return [validate_manifest(path) for path in paths]
