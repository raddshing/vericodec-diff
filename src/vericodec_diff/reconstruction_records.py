from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from vericodec_diff.real_images import REAL_IMAGE_MANIFEST_COLUMNS, load_real_manifest
from vericodec_diff.synthetic_dataset import MANIFEST_COLUMNS as SYNTHETIC_MANIFEST_COLUMNS


RECONSTRUCTION_RECORD_COLUMNS = (
    "sample_id",
    "image_path",
    "split",
    "category",
    "source_name",
)
SPLIT_ORDER = ("train", "val", "test")
DEFAULT_SPLIT_FRACTIONS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}
SYNTHETIC_SOURCE_NAME = "synthetic"


class ReconstructionRecordValidationError(ValueError):
    """Raised when reconstruction-record inputs or outputs violate the local schema."""


@dataclass(frozen=True)
class ReconstructionCandidate:
    sample_id: str
    image_path: str
    category: str
    source_name: str


def resolve_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path


def display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def deep_update(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in dict(overrides).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_update(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def default_reconstruction_record_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "synthetic_manifest_path": "data/manifests/synthetic_manifest.csv",
            "real_manifest_path": "data/manifests/real_images_manifest.csv",
            "records_path": "data/manifests/reconstruction_records.csv",
            "output_dir": "outputs/reconstruction_records",
        },
        "splits": {
            "fractions": deepcopy(DEFAULT_SPLIT_FRACTIONS),
        },
    }


def _load_manifest_rows(path: Path, *, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != tuple(expected_columns):
            raise ReconstructionRecordValidationError(
                f"{path.name}: expected columns {tuple(expected_columns)}, found {fieldnames}"
            )
        return [dict(row) for row in reader]


def _resolve_fraction_mapping(raw_value: Mapping[str, Any]) -> dict[str, float]:
    fractions: dict[str, float] = {}
    for split in SPLIT_ORDER:
        if split not in raw_value:
            raise ReconstructionRecordValidationError(
                f"splits.fractions must define {SPLIT_ORDER}, missing {split!r}"
            )
        value = float(raw_value[split])
        if value < 0.0 or value > 1.0:
            raise ReconstructionRecordValidationError(f"splits.fractions[{split!r}] must be between 0 and 1")
        fractions[split] = value

    extra_keys = sorted(set(raw_value) - set(SPLIT_ORDER))
    if extra_keys:
        raise ReconstructionRecordValidationError(f"splits.fractions contains unsupported keys {extra_keys}")

    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ReconstructionRecordValidationError("splits.fractions must sum to exactly 1.0")
    return fractions


def resolve_reconstruction_record_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_reconstruction_record_config(), raw_config)
    paths = dict(config.get("paths", {}))
    splits = dict(config.get("splits", {}))

    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()
    synthetic_manifest_path = resolve_path(
        resolved_repo_root,
        str(paths.get("synthetic_manifest_path", "data/manifests/synthetic_manifest.csv")),
    ).resolve()
    real_manifest_path = resolve_path(
        resolved_repo_root,
        str(paths.get("real_manifest_path", "data/manifests/real_images_manifest.csv")),
    ).resolve()
    records_path = resolve_path(
        resolved_repo_root,
        str(paths.get("records_path", "data/manifests/reconstruction_records.csv")),
    ).resolve()
    output_dir = resolve_path(
        resolved_repo_root,
        str(paths.get("output_dir", "outputs/reconstruction_records")),
    ).resolve()

    for name, path in {
        "synthetic_manifest_path": synthetic_manifest_path,
        "real_manifest_path": real_manifest_path,
        "records_path": records_path,
        "output_dir": output_dir,
    }.items():
        try:
            path.relative_to(resolved_repo_root)
        except ValueError as exc:
            raise ReconstructionRecordValidationError(
                f"{name} must resolve inside repo_root for stable local paths"
            ) from exc

    if not synthetic_manifest_path.is_file():
        raise FileNotFoundError(f"Synthetic manifest not found: {synthetic_manifest_path}")
    if not real_manifest_path.is_file():
        raise FileNotFoundError(f"Real-image manifest not found: {real_manifest_path}")

    fractions = _resolve_fraction_mapping(dict(splits.get("fractions", DEFAULT_SPLIT_FRACTIONS)))

    return {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "synthetic_manifest_path": str(synthetic_manifest_path),
            "real_manifest_path": str(real_manifest_path),
            "records_path": str(records_path),
            "output_dir": str(output_dir),
        },
        "splits": {
            "fractions": fractions,
        },
    }


def load_reconstruction_candidates(config: Mapping[str, Any]) -> list[ReconstructionCandidate]:
    repo_root = Path(config["paths"]["repo_root"])
    synthetic_manifest_path = Path(config["paths"]["synthetic_manifest_path"])
    real_manifest_path = Path(config["paths"]["real_manifest_path"])

    synthetic_rows = _load_manifest_rows(synthetic_manifest_path, expected_columns=SYNTHETIC_MANIFEST_COLUMNS)
    real_rows = load_real_manifest(real_manifest_path)

    sample_ids: set[str] = set()
    candidates: list[ReconstructionCandidate] = []

    for row in synthetic_rows:
        candidate = ReconstructionCandidate(
            sample_id=row["image_id"],
            image_path=row["local_path"],
            category=row["category"],
            source_name=SYNTHETIC_SOURCE_NAME,
        )
        _validate_candidate(candidate, repo_root=repo_root)
        if candidate.sample_id in sample_ids:
            raise ReconstructionRecordValidationError(f"Duplicate sample_id across manifests: {candidate.sample_id!r}")
        sample_ids.add(candidate.sample_id)
        candidates.append(candidate)

    for row in real_rows:
        candidate = ReconstructionCandidate(
            sample_id=row["image_id"],
            image_path=row["local_path"],
            category=row["category"],
            source_name=row["source_name"],
        )
        _validate_candidate(candidate, repo_root=repo_root)
        if candidate.sample_id in sample_ids:
            raise ReconstructionRecordValidationError(f"Duplicate sample_id across manifests: {candidate.sample_id!r}")
        sample_ids.add(candidate.sample_id)
        candidates.append(candidate)

    if not candidates:
        raise ReconstructionRecordValidationError("No samples were available to build reconstruction records")
    return candidates


def _validate_candidate(candidate: ReconstructionCandidate, *, repo_root: Path) -> None:
    if not candidate.sample_id.strip():
        raise ReconstructionRecordValidationError("sample_id must be non-empty")
    if not candidate.category.strip():
        raise ReconstructionRecordValidationError(f"{candidate.sample_id}: category must be non-empty")
    if not candidate.source_name.strip():
        raise ReconstructionRecordValidationError(f"{candidate.sample_id}: source_name must be non-empty")

    image_path = Path(candidate.image_path)
    if image_path.is_absolute():
        raise ReconstructionRecordValidationError(f"{candidate.sample_id}: image_path must stay relative")
    resolved_image_path = (repo_root / image_path).resolve()
    if not resolved_image_path.is_file():
        raise FileNotFoundError(f"{candidate.sample_id}: missing source image {resolved_image_path}")


def _stable_assignment_key(candidate: ReconstructionCandidate) -> str:
    payload = f"{candidate.category}|{candidate.source_name}|{candidate.sample_id}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _compute_split_counts(category_size: int, fractions: Mapping[str, float]) -> dict[str, int]:
    raw_targets = {
        split: float(fractions[split]) * category_size
        for split in SPLIT_ORDER
    }
    counts = {
        split: int(math.floor(raw_targets[split]))
        for split in SPLIT_ORDER
    }
    remaining = category_size - sum(counts.values())
    if remaining < 0:
        raise ReconstructionRecordValidationError("Split allocation underflowed the category size")

    ranked_splits = sorted(
        SPLIT_ORDER,
        key=lambda split: (-(raw_targets[split] - counts[split]), SPLIT_ORDER.index(split)),
    )
    for split in ranked_splits[:remaining]:
        counts[split] += 1

    if sum(counts.values()) != category_size:
        raise ReconstructionRecordValidationError("Split allocation did not preserve the category size")
    return counts


def assign_stratified_splits(
    candidates: Sequence[ReconstructionCandidate],
    *,
    fractions: Mapping[str, float],
) -> list[dict[str, str]]:
    grouped: dict[str, list[ReconstructionCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.category].append(candidate)

    rows: list[dict[str, str]] = []
    for category in sorted(grouped):
        ordered_candidates = sorted(
            grouped[category],
            key=lambda candidate: (_stable_assignment_key(candidate), candidate.sample_id),
        )
        split_counts = _compute_split_counts(len(ordered_candidates), fractions)

        offset = 0
        for split in SPLIT_ORDER:
            stop = offset + split_counts[split]
            for candidate in ordered_candidates[offset:stop]:
                rows.append(
                    {
                        "sample_id": candidate.sample_id,
                        "image_path": candidate.image_path,
                        "split": split,
                        "category": candidate.category,
                        "source_name": candidate.source_name,
                    }
                )
            offset = stop

    rows.sort(
        key=lambda row: (
            SPLIT_ORDER.index(row["split"]),
            row["category"],
            row["source_name"],
            row["sample_id"],
        )
    )
    return rows


def write_reconstruction_records(rows: Sequence[Mapping[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECONSTRUCTION_RECORD_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in RECONSTRUCTION_RECORD_COLUMNS})


def build_reconstruction_records(config: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    records_path = Path(config["paths"]["records_path"])
    fractions = config["splits"]["fractions"]

    candidates = load_reconstruction_candidates(config)
    rows = assign_stratified_splits(candidates, fractions=fractions)
    write_reconstruction_records(rows, records_path)

    return {
        "record_count": len(rows),
        "records_path": str(records_path),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "source_counts": dict(Counter(row["source_name"] for row in rows)),
        "sample_paths": [row["image_path"] for row in rows[:3]],
        "records_path_display": display_path(records_path, repo_root),
    }


def validate_reconstruction_records(
    path: str | Path,
    *,
    repo_root: Path,
    verify_files: bool = False,
) -> dict[str, Any]:
    records_path = Path(path)
    rows = _load_manifest_rows(records_path, expected_columns=RECONSTRUCTION_RECORD_COLUMNS)

    sample_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for row_number, row in enumerate(rows, start=2):
        for column in RECONSTRUCTION_RECORD_COLUMNS:
            value = row.get(column, "")
            if value is None or not str(value).strip():
                raise ReconstructionRecordValidationError(
                    f"{records_path.name}: row {row_number} has empty value for {column}"
                )

        if row["split"] not in SPLIT_ORDER:
            raise ReconstructionRecordValidationError(
                f"{records_path.name}: row {row_number} has unsupported split {row['split']!r}"
            )
        if row["sample_id"] in sample_ids:
            raise ReconstructionRecordValidationError(
                f"{records_path.name}: duplicate sample_id {row['sample_id']!r}"
            )
        sample_ids.add(row["sample_id"])

        image_path = Path(row["image_path"])
        if image_path.is_absolute():
            raise ReconstructionRecordValidationError(
                f"{records_path.name}: row {row_number} image_path must stay relative"
            )

        category_counts[row["category"]] += 1
        split_counts[row["split"]] += 1
        source_counts[row["source_name"]] += 1

        if verify_files:
            resolved_image_path = (repo_root / image_path).resolve()
            if not resolved_image_path.is_file():
                raise ReconstructionRecordValidationError(
                    f"{records_path.name}: row {row_number} points to missing image {row['image_path']!r}"
                )

    return {
        "manifest": records_path.name,
        "total": len(rows),
        "category_counts": dict(category_counts),
        "split_counts": dict(split_counts),
        "source_counts": dict(source_counts),
    }
