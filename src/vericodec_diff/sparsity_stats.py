from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import ndimage

from vericodec_diff.patch_metrics import (
    DEFAULT_GRID_SIZE,
    DEFAULT_PATCH_SIZE,
    PATCH_COUNT,
    PATCH_ERROR_SUFFIX,
    PATCH_METRIC_NAMES,
    SampleDiscoveryError,
    deep_update,
    display_path,
    parse_csv_items,
    resolve_path,
    validate_patch_metric_array,
)
from vericodec_diff.patch_error_targets import (
    SUPPORTED_SPARSITY_METRIC_NAMES,
    patch_metric_source_command,
)


SPARSITY_SUMMARY_VERSION = 1
_ROOK_STRUCTURE = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)


def default_sparsity_stats_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "error_map_root": "outputs/error_maps",
            "output_root": "outputs/metrics",
            "phase": "kill_test",
        },
        "data": {
            "split_filter": [],
            "limit": None,
            "metric_names": list(PATCH_METRIC_NAMES),
        },
        "stats": {
            "concentration_percents": [5, 10, 15, 20],
        },
    }


def resolve_sparsity_stats_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_sparsity_stats_config(), raw_config)
    paths = dict(config.get("paths", {}))
    data = dict(config.get("data", {}))
    stats = dict(config.get("stats", {}))

    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()
    error_map_root = resolve_path(
        resolved_repo_root,
        str(paths.get("error_map_root", "outputs/error_maps")),
    ).resolve()
    output_root = resolve_path(
        resolved_repo_root,
        str(paths.get("output_root", "outputs/metrics")),
    ).resolve()
    phase = str(paths.get("phase", "kill_test")).strip() or "kill_test"
    output_dir = (output_root / phase).resolve()

    for name, path in {
        "error_map_root": error_map_root,
        "output_root": output_root,
        "output_dir": output_dir,
    }.items():
        try:
            path.relative_to(resolved_repo_root)
        except ValueError as exc:
            raise ValueError(f"{name} must resolve inside repo_root for stable local paths") from exc

    split_filter = parse_csv_items(data.get("split_filter"))
    limit_value = data.get("limit")
    limit = None if limit_value in (None, "") else int(limit_value)
    if limit is not None and limit <= 0:
        raise ValueError("data.limit must be a positive integer when provided")

    metric_names = parse_csv_items(data.get("metric_names"))
    if not metric_names:
        metric_names = list(PATCH_METRIC_NAMES)
    if not metric_names:
        raise ValueError("data.metric_names must contain at least one metric name")
    unsupported_metrics = [name for name in metric_names if name not in SUPPORTED_SPARSITY_METRIC_NAMES]
    if unsupported_metrics:
        raise ValueError(f"Unsupported patch metrics requested: {unsupported_metrics}")

    concentration_percents = [int(value) for value in stats.get("concentration_percents", [5, 10, 15, 20])]
    if not concentration_percents:
        raise ValueError("stats.concentration_percents must not be empty")
    for value in concentration_percents:
        if value <= 0 or value > 100:
            raise ValueError("stats.concentration_percents values must lie within [1, 100]")

    return {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "error_map_root": str(error_map_root),
            "output_root": str(output_root),
            "output_dir": str(output_dir),
            "phase": phase,
        },
        "data": {
            "split_filter": split_filter,
            "limit": limit,
            "metric_names": metric_names,
        },
        "stats": {
            "concentration_percents": concentration_percents,
        },
    }


def _scalar_string(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        raise ValueError("Expected a scalar string payload")
    return str(value)


def _scalar_int(value: Any) -> int:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return int(value.item())
        raise ValueError("Expected a scalar integer payload")
    return int(value)


def discover_patch_error_paths(config: Mapping[str, Any]) -> list[Path]:
    error_map_root = Path(config["paths"]["error_map_root"])
    if not error_map_root.exists():
        raise FileNotFoundError(f"Patch-error root not found: {error_map_root}")

    split_filter = set(config["data"]["split_filter"])
    paths: list[Path] = []
    for path in sorted(error_map_root.rglob(f"*{PATCH_ERROR_SUFFIX}")):
        split = path.parent.name
        if split_filter and split not in split_filter:
            continue
        paths.append(path)

    limit = config["data"]["limit"]
    if limit is not None:
        paths = paths[: int(limit)]
    if not paths:
        raise SampleDiscoveryError("No patch-error NPZ files matched the current selection")
    return paths


def load_patch_error_npz(path: Path, *, metric_names: Sequence[str]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        patch_size = _scalar_int(payload["patch_size"])
        grid_height = _scalar_int(payload["grid_height"])
        grid_width = _scalar_int(payload["grid_width"])
        if patch_size != DEFAULT_PATCH_SIZE:
            raise ValueError(f"{path}: expected patch_size {DEFAULT_PATCH_SIZE}, found {patch_size}")
        if (grid_height, grid_width) != (DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE):
            raise ValueError(
                f"{path}: expected patch grid {(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)}, found {(grid_height, grid_width)}"
            )

        metrics: dict[str, np.ndarray] = {}
        for name in metric_names:
            if name not in payload.files:
                raise ValueError(f"{path}: missing patch metric {name!r}; run {patch_metric_source_command(name)} first")
            metrics[name] = validate_patch_metric_array(payload[name], metric_name=f"{path.name}:{name}")
        return {
            "sample_id": _scalar_string(payload["sample_id"]),
            "split": _scalar_string(payload["split"]),
            "lpips_backend": _scalar_string(payload["lpips_backend"]),
            "wavelet": _scalar_string(payload["wavelet"]),
            "metrics": metrics,
        }


def concentration_at_percent(values: np.ndarray, percent: int) -> float:
    array = validate_patch_metric_array(values, metric_name="concentration")
    total = float(array.sum(dtype=np.float64))
    if total <= 0.0:
        return 0.0
    k = max(1, int(np.ceil(array.size * (float(percent) / 100.0))))
    indices = np.argsort(-array, kind="mergesort")[:k]
    return float(array[indices].sum(dtype=np.float64) / total)


def gini_coefficient(values: np.ndarray) -> float:
    array = np.sort(validate_patch_metric_array(values, metric_name="gini"))
    total = float(array.sum(dtype=np.float64))
    if total <= 0.0:
        return 0.0
    index = np.arange(1, array.size + 1, dtype=np.float64)
    numerator = np.sum((2.0 * index - array.size - 1.0) * array, dtype=np.float64)
    return float(numerator / (array.size * total))


def morans_i(values: np.ndarray) -> float:
    array = validate_patch_metric_array(values, metric_name="morans_i").reshape(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
    centered = array - float(array.mean(dtype=np.float64))
    denominator = float(np.sum(np.square(centered), dtype=np.float64))
    if denominator <= 0.0:
        return 0.0

    horizontal_products = centered[:, :-1] * centered[:, 1:]
    vertical_products = centered[:-1, :] * centered[1:, :]
    numerator = 2.0 * float(horizontal_products.sum(dtype=np.float64) + vertical_products.sum(dtype=np.float64))
    total_weights = 2.0 * float(horizontal_products.size + vertical_products.size)
    return float((PATCH_COUNT / total_weights) * (numerator / denominator))


def topk_mask(values: np.ndarray, percent: int) -> np.ndarray:
    array = validate_patch_metric_array(values, metric_name="topk_mask")
    if float(array.sum(dtype=np.float64)) <= 0.0:
        return np.zeros_like(array, dtype=bool)
    positive_indices = np.flatnonzero(array > 0.0)
    if positive_indices.size == 0:
        return np.zeros_like(array, dtype=bool)
    k = max(1, int(np.ceil(array.size * (float(percent) / 100.0))))
    positive_values = array[positive_indices]
    order = np.argsort(-positive_values, kind="mergesort")[: min(k, positive_indices.size)]
    indices = positive_indices[order]
    mask = np.zeros(array.size, dtype=bool)
    mask[indices] = True
    return mask


def connected_component_summary(values: np.ndarray, percent: int) -> dict[str, float | int]:
    mask = topk_mask(values, percent).reshape(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
    if not mask.any():
        return {
            "active_patch_count": 0,
            "component_count": 0,
            "largest_component_size": 0,
            "mean_component_size": 0.0,
        }

    labels, component_count = ndimage.label(mask, structure=_ROOK_STRUCTURE)
    sizes = np.bincount(labels.ravel())[1:]
    return {
        "active_patch_count": int(mask.sum(dtype=np.int64)),
        "component_count": int(component_count),
        "largest_component_size": int(sizes.max(initial=0)),
        "mean_component_size": float(sizes.mean(dtype=np.float64)) if sizes.size else 0.0,
    }


def _round_float(value: float) -> float:
    return float(round(float(value), 8))


def build_sample_stat_record(
    *,
    sample_id: str,
    split: str,
    metric_name: str,
    values: np.ndarray,
    concentration_percents: Sequence[int],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "metric": metric_name,
        "gini": _round_float(gini_coefficient(values)),
        "morans_i": _round_float(morans_i(values)),
    }
    for percent in concentration_percents:
        prefix = f"pct_{percent:02d}"
        record[f"concentration_at_{prefix}"] = _round_float(concentration_at_percent(values, percent))
        component_summary = connected_component_summary(values, percent)
        record[f"active_patches_at_{prefix}"] = int(component_summary["active_patch_count"])
        record[f"components_at_{prefix}"] = int(component_summary["component_count"])
        record[f"largest_component_at_{prefix}"] = int(component_summary["largest_component_size"])
        record[f"mean_component_size_at_{prefix}"] = _round_float(component_summary["mean_component_size"])
    return record


def _aggregate_records(records: Sequence[Mapping[str, Any]], concentration_percents: Sequence[int]) -> dict[str, Any]:
    if not records:
        return {}

    summary: dict[str, Any] = {
        "sample_count": len(records),
        "mean_gini": _round_float(np.mean([float(record["gini"]) for record in records], dtype=np.float64)),
        "mean_morans_i": _round_float(
            np.mean([float(record["morans_i"]) for record in records], dtype=np.float64)
        ),
    }
    for percent in concentration_percents:
        prefix = f"pct_{percent:02d}"
        summary[f"mean_concentration_at_{prefix}"] = _round_float(
            np.mean([float(record[f"concentration_at_{prefix}"]) for record in records], dtype=np.float64)
        )
        summary[f"mean_active_patches_at_{prefix}"] = _round_float(
            np.mean([float(record[f"active_patches_at_{prefix}"]) for record in records], dtype=np.float64)
        )
        summary[f"mean_components_at_{prefix}"] = _round_float(
            np.mean([float(record[f"components_at_{prefix}"]) for record in records], dtype=np.float64)
        )
        summary[f"mean_largest_component_at_{prefix}"] = _round_float(
            np.mean([float(record[f"largest_component_at_{prefix}"]) for record in records], dtype=np.float64)
        )
        summary[f"mean_component_size_at_{prefix}"] = _round_float(
            np.mean([float(record[f"mean_component_size_at_{prefix}"]) for record in records], dtype=np.float64)
        )
    return summary


def _group_records(
    records: Sequence[Mapping[str, Any]],
    *,
    by: str,
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        key = str(record[by])
        grouped.setdefault(key, []).append(record)
    return grouped


def compute_sparsity_stats(config: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    patch_error_paths = discover_patch_error_paths(config)
    concentration_percents = list(config["stats"]["concentration_percents"])
    metric_names = list(config["data"]["metric_names"])

    sample_records: list[dict[str, Any]] = []
    lpips_backends: set[str] = set()
    wavelets: set[str] = set()
    split_counts: dict[str, int] = {}

    for path in patch_error_paths:
        payload = load_patch_error_npz(path, metric_names=metric_names)
        lpips_backends.add(payload["lpips_backend"])
        wavelets.add(payload["wavelet"])
        split_counts[payload["split"]] = split_counts.get(payload["split"], 0) + 1
        for metric_name in metric_names:
            record = build_sample_stat_record(
                sample_id=payload["sample_id"],
                split=payload["split"],
                metric_name=metric_name,
                values=payload["metrics"][metric_name],
                concentration_percents=concentration_percents,
            )
            sample_records.append(record)

    sample_records.sort(key=lambda item: (str(item["split"]), str(item["sample_id"]), str(item["metric"])))

    metric_summaries = {
        metric_name: _aggregate_records(
            [record for record in sample_records if record["metric"] == metric_name],
            concentration_percents,
        )
        for metric_name in metric_names
    }

    split_metric_summaries: dict[str, dict[str, Any]] = {}
    for split, split_records in sorted(_group_records(sample_records, by="split").items()):
        split_metric_summaries[split] = {
            metric_name: _aggregate_records(
                [record for record in split_records if record["metric"] == metric_name],
                concentration_percents,
            )
            for metric_name in metric_names
        }

    return {
        "summary": {
            "summary_version": SPARSITY_SUMMARY_VERSION,
            "phase": config["paths"]["phase"],
            "error_map_root": display_path(Path(config["paths"]["error_map_root"]), repo_root),
            "input_file_count": len(patch_error_paths),
            "sample_count": len({(record["split"], record["sample_id"]) for record in sample_records}),
            "split_counts": split_counts,
            "metric_names": metric_names,
            "concentration_percents": concentration_percents,
            "lpips_backends": sorted(lpips_backends),
            "wavelets": sorted(wavelets),
            "metric_summaries": metric_summaries,
            "split_metric_summaries": split_metric_summaries,
        },
        "records": sample_records,
    }


def write_sparsity_records_csv(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    concentration_percents: Iterable[int],
) -> None:
    concentration_percents = list(concentration_percents)
    fieldnames = ["sample_id", "split", "metric", "gini", "morans_i"]
    for percent in concentration_percents:
        prefix = f"pct_{percent:02d}"
        fieldnames.extend(
            [
                f"concentration_at_{prefix}",
                f"active_patches_at_{prefix}",
                f"components_at_{prefix}",
                f"largest_component_at_{prefix}",
                f"mean_component_size_at_{prefix}",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})
