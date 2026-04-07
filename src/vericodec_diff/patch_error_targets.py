from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vericodec_diff.patch_metrics import (
    DEFAULT_GRID_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PATCH_SIZE,
    PATCH_ERROR_SUFFIX,
    PATCH_METRIC_NAMES,
    SampleDiscoveryError,
    deep_update,
    display_path,
    parse_csv_items,
    resolve_path,
    validate_patch_metric_array,
    write_patch_error_npz,
)


PATCH_ERROR_METRIC_NAME = "patch_error"
PATCH_LABEL_TOP15_NAME = "patch_label_top15"
PATCH_LABEL_TOP15_PERCENT = 15
PATCH_ERROR_TARGETS_SUMMARY_VERSION = 1
DEFAULT_PATCH_ERROR_TARGETS_OUTPUT_DIR = "outputs/metrics/kill_test/patch_error_targets"
SUPPORTED_SPARSITY_METRIC_NAMES = PATCH_METRIC_NAMES + (PATCH_ERROR_METRIC_NAME,)
SUPPORTED_VERIFIER_LABEL_METRIC_NAMES = SUPPORTED_SPARSITY_METRIC_NAMES + (PATCH_LABEL_TOP15_NAME,)


def default_patch_error_target_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "error_map_root": "outputs/error_maps",
            "output_dir": DEFAULT_PATCH_ERROR_TARGETS_OUTPUT_DIR,
        },
        "data": {
            "split_filter": [],
            "limit": None,
        },
    }


def resolve_patch_error_target_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_patch_error_target_config(), raw_config)
    paths = dict(config.get("paths", {}))
    data = dict(config.get("data", {}))

    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()
    error_map_root = resolve_path(
        resolved_repo_root,
        str(paths.get("error_map_root", "outputs/error_maps")),
    ).resolve()
    output_dir = resolve_path(
        resolved_repo_root,
        str(paths.get("output_dir", DEFAULT_PATCH_ERROR_TARGETS_OUTPUT_DIR)),
    ).resolve()

    for name, path in {
        "error_map_root": error_map_root,
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

    return {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "error_map_root": str(error_map_root),
            "output_dir": str(output_dir),
        },
        "data": {
            "split_filter": split_filter,
            "limit": limit,
        },
    }


def patch_metric_source_command(metric_name: str) -> str:
    if metric_name in PATCH_METRIC_NAMES:
        return "scripts/compute_patch_errors.py"
    if metric_name in {PATCH_ERROR_METRIC_NAME, PATCH_LABEL_TOP15_NAME}:
        return "scripts/augment_patch_error_targets.py"
    raise ValueError(f"Unsupported patch metric {metric_name!r}")


def discover_patch_error_target_paths(config: Mapping[str, Any]) -> list[Path]:
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


def compute_patch_error_metric(
    lpips: np.ndarray,
    one_minus_ssim: np.ndarray,
    hf_wavelet_l1: np.ndarray,
) -> np.ndarray:
    lpips_values = validate_patch_metric_array(lpips, metric_name="lpips")
    one_minus_ssim_values = validate_patch_metric_array(one_minus_ssim, metric_name="one_minus_ssim")
    hf_wavelet_l1_values = validate_patch_metric_array(hf_wavelet_l1, metric_name="hf_wavelet_l1")
    patch_error = (
        0.5 * lpips_values.astype(np.float64)
        + 0.25 * one_minus_ssim_values.astype(np.float64)
        + 0.25 * hf_wavelet_l1_values.astype(np.float64)
    ).astype(np.float32)
    return validate_patch_metric_array(patch_error, metric_name=PATCH_ERROR_METRIC_NAME)


def compute_top_percent_patch_labels(values: np.ndarray, *, percent: int = PATCH_LABEL_TOP15_PERCENT) -> np.ndarray:
    if percent <= 0 or percent > 100:
        raise ValueError("percent must lie within [1, 100]")
    metric_values = validate_patch_metric_array(values, metric_name=PATCH_LABEL_TOP15_NAME)
    selected_patch_count = max(1, int(np.ceil(metric_values.size * (float(percent) / 100.0))))
    selected_indices = np.argsort(-metric_values, kind="mergesort")[:selected_patch_count]
    labels = np.zeros(metric_values.size, dtype=np.uint8)
    labels[selected_indices] = 1
    return labels


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


def _payload_names(payload: Mapping[str, Any]) -> Sequence[str]:
    files = getattr(payload, "files", None)
    if files is not None:
        return [str(name) for name in files]
    return [str(name) for name in payload.keys()]


def augment_patch_error_payload(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    patch_size = _scalar_int(payload["patch_size"])
    image_size = _scalar_int(payload["image_size"])
    grid_height = _scalar_int(payload["grid_height"])
    grid_width = _scalar_int(payload["grid_width"])
    if patch_size != DEFAULT_PATCH_SIZE:
        raise ValueError(f"expected patch_size {DEFAULT_PATCH_SIZE}, found {patch_size}")
    if image_size != DEFAULT_IMAGE_SIZE:
        raise ValueError(f"expected image_size {DEFAULT_IMAGE_SIZE}, found {image_size}")
    if (grid_height, grid_width) != (DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE):
        raise ValueError(f"expected grid {(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)}, found {(grid_height, grid_width)}")

    output = {name: np.asarray(payload[name]) for name in _payload_names(payload)}
    for metric_name in PATCH_METRIC_NAMES:
        if metric_name not in output:
            raise ValueError(
                f"missing patch metric {metric_name!r}; run {patch_metric_source_command(metric_name)} first"
            )

    patch_error = compute_patch_error_metric(
        output["lpips"],
        output["one_minus_ssim"],
        output["hf_wavelet_l1"],
    )
    patch_label_top15 = compute_top_percent_patch_labels(patch_error)
    output[PATCH_ERROR_METRIC_NAME] = patch_error
    output[PATCH_LABEL_TOP15_NAME] = patch_label_top15
    return output


def _atomic_rewrite_patch_error_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    temp_path = path.with_name(f"{path.stem}__tmp{path.suffix}")
    write_patch_error_npz(temp_path, payload)
    temp_path.replace(path)


def augment_patch_error_targets(config: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    paths = discover_patch_error_target_paths(config)

    split_counts = Counter(path.parent.name for path in paths)
    records: list[dict[str, Any]] = []
    updated_file_count = 0

    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            augmented_payload = augment_patch_error_payload(payload)
            sample_id = _scalar_string(payload["sample_id"])
            split = _scalar_string(payload["split"])

        _atomic_rewrite_patch_error_npz(path, augmented_payload)
        updated_file_count += 1
        label_values = np.asarray(augmented_payload[PATCH_LABEL_TOP15_NAME], dtype=np.uint8).reshape(-1)
        records.append(
            {
                "sample_id": sample_id,
                "split": split,
                "error_map_path": display_path(path, repo_root),
                "patch_error_mean": round(float(np.mean(augmented_payload[PATCH_ERROR_METRIC_NAME], dtype=np.float64)), 8),
                "positive_patch_count": int(label_values.sum(dtype=np.int64)),
                "status": "augmented",
            }
        )

    return {
        "summary": {
            "version": PATCH_ERROR_TARGETS_SUMMARY_VERSION,
            "error_map_root": display_path(Path(config["paths"]["error_map_root"]), repo_root),
            "continuous_metric_name": PATCH_ERROR_METRIC_NAME,
            "label_metric_name": PATCH_LABEL_TOP15_NAME,
            "label_percent": PATCH_LABEL_TOP15_PERCENT,
            "input_file_count": len(paths),
            "updated_file_count": updated_file_count,
            "split_counts": dict(split_counts),
        },
        "records": records,
    }


def write_patch_error_target_records_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "sample_id",
        "split",
        "error_map_path",
        "patch_error_mean",
        "positive_patch_count",
        "status",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})
