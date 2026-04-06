from __future__ import annotations

import csv
import io
import json
import pickle
from collections.abc import Mapping, Sequence
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage import filters

from vericodec_diff.patch_metrics import (
    DEFAULT_DC_AE_MODEL_ID,
    DEFAULT_GRID_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PATCH_SIZE,
    DCAEReconstructor,
    MetricBackendUnavailableError,
    PATCH_COUNT,
    PatchMetricValidationError,
    compute_patchwise_hf_wavelet_l1,
    deep_update,
    display_path,
    image_to_float_array,
    load_rgb_image,
    parse_csv_items,
    patchify_image_array,
    resolve_path,
    validate_patch_metric_array,
)


VERIFIER_FEATURE_VERSION = 1
SIGNAL_SUFFIX = "__signals.npz"
INDEX_FILENAME = "index.csv"
DEFAULT_ENTROPY_BINS = 32
DEFAULT_JPEG_QUALITY = 95
DEFAULT_SIGNAL_NAMES = (
    "decode_reencode_consistency",
    "late_step_update_magnitude",
    "wavelet_instability",
    "sobel_edge_magnitude",
    "local_intensity_variance",
    "patch_entropy",
)
OPTIONAL_SIGNAL_NAMES = ("cross_attention_instability",)
ALL_SIGNAL_NAMES = DEFAULT_SIGNAL_NAMES + OPTIONAL_SIGNAL_NAMES
TRACE_REQUIRED_SIGNALS = {
    "late_step_update_magnitude",
    "wavelet_instability",
    "cross_attention_instability",
}


class VerifierFeatureDiscoveryError(ValueError):
    """Raised when the feature extractor cannot deterministically interpret input records."""


class SignalUnavailableError(RuntimeError):
    """Raised when a requested signal cannot be computed for the current sample."""


@dataclass(frozen=True)
class VerifierSampleRecord:
    sample_id: str
    split: str
    image_path: Path
    trace_path: Path | None
    trace_status: str
    image_local_path: str
    trace_local_path: str
    metadata: dict[str, str]


class _BaseDecodeReencodeBackend:
    backend_name = "undefined"

    def reconstruct(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError

    def close(self) -> None:
        return None


class DCAEDecodeReencodeBackend(_BaseDecodeReencodeBackend):
    backend_name = "diffusers:AutoencoderDC"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.reconstructor = DCAEReconstructor(
            {
                "reconstruction": {
                    "model_id": str(config["decode_reencode"]["model_id"]),
                    "device": str(config["decode_reencode"]["device"]),
                    "torch_dtype": str(config["decode_reencode"]["torch_dtype"]),
                    "local_files_only": bool(config["decode_reencode"]["local_files_only"]),
                }
            }
        )

    def reconstruct(self, image: Image.Image) -> Image.Image:
        return self.reconstructor.reconstruct(image)

    def close(self) -> None:
        self.reconstructor.close()


class JpegDecodeReencodeBackend(_BaseDecodeReencodeBackend):
    def __init__(self, *, quality: int) -> None:
        self.quality = int(quality)
        self.backend_name = f"pil:jpeg_q{self.quality}"

    def reconstruct(self, image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=self.quality,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as roundtrip:
            return roundtrip.convert("RGB")


def default_verifier_feature_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "records_csv": None,
            "feature_root": "data/processed/verifier_features",
        },
        "data": {
            "image_size": DEFAULT_IMAGE_SIZE,
            "patch_size": DEFAULT_PATCH_SIZE,
            "sample_id_column": None,
            "image_path_column": None,
            "trace_path_column": None,
            "trace_status_column": None,
            "split_column": "split",
            "default_split": "unspecified",
            "split_filter": [],
            "limit": None,
        },
        "decode_reencode": {
            "backend": "dc_ae",
            "model_id": DEFAULT_DC_AE_MODEL_ID,
            "device": "cuda",
            "torch_dtype": "float32",
            "local_files_only": False,
            "jpeg_quality": DEFAULT_JPEG_QUALITY,
        },
        "trace": {
            "wavelet": "haar",
            "max_images": None,
        },
        "signals": {
            "enabled": list(DEFAULT_SIGNAL_NAMES),
            "optional": list(OPTIONAL_SIGNAL_NAMES),
            "cross_attention_instability": False,
        },
        "heuristics": {
            "entropy_bins": DEFAULT_ENTROPY_BINS,
        },
        "runtime": {
            "overwrite": False,
            "allow_missing_optional_signals": True,
        },
    }


def _infer_column(
    fieldnames: Sequence[str],
    configured: str | None,
    candidates: Sequence[str],
    *,
    required: bool,
    label: str,
) -> str | None:
    if configured not in (None, ""):
        if configured not in fieldnames:
            raise VerifierFeatureDiscoveryError(
                f"Configured {label} {configured!r} is missing from the records CSV"
            )
        return configured
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    if required:
        raise VerifierFeatureDiscoveryError(
            f"Could not infer {label}; expected one of {tuple(candidates)} in the records CSV"
        )
    return None


def _resolve_record_path(raw_path: str, repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _validate_signal_names(names: Sequence[str], *, label: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_name in names:
        signal_name = str(raw_name).strip()
        if not signal_name:
            continue
        if signal_name not in ALL_SIGNAL_NAMES:
            raise ValueError(f"{label} contains unsupported signal {signal_name!r}")
        if signal_name not in seen:
            normalized.append(signal_name)
            seen.add(signal_name)
    return normalized


def resolve_verifier_feature_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_verifier_feature_config(), raw_config)
    paths = dict(config.get("paths", {}))
    data = dict(config.get("data", {}))
    decode_reencode = dict(config.get("decode_reencode", {}))
    trace = dict(config.get("trace", {}))
    signals = dict(config.get("signals", {}))
    runtime = dict(config.get("runtime", {}))

    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()

    records_csv_raw = paths.get("records_csv")
    if records_csv_raw in (None, ""):
        raise ValueError("paths.records_csv is required")
    records_csv = resolve_path(resolved_repo_root, str(records_csv_raw)).resolve()
    if not records_csv.is_file():
        raise FileNotFoundError(f"Feature-record CSV not found: {records_csv}")

    feature_root = resolve_path(
        resolved_repo_root,
        str(paths.get("feature_root", "data/processed/verifier_features")),
    ).resolve()
    try:
        feature_root.relative_to(resolved_repo_root)
    except ValueError as exc:
        raise ValueError("paths.feature_root must resolve inside repo_root for stable paths") from exc

    image_size = int(data.get("image_size", DEFAULT_IMAGE_SIZE))
    patch_size = int(data.get("patch_size", DEFAULT_PATCH_SIZE))
    if image_size != DEFAULT_IMAGE_SIZE:
        raise ValueError(
            f"Verifier features are locked to {DEFAULT_IMAGE_SIZE}x{DEFAULT_IMAGE_SIZE} images"
        )
    if patch_size != DEFAULT_PATCH_SIZE:
        raise ValueError(f"Verifier features are locked to {DEFAULT_PATCH_SIZE}x{DEFAULT_PATCH_SIZE} patches")

    split_filter = parse_csv_items(data.get("split_filter"))
    limit_value = data.get("limit")
    limit = None if limit_value in (None, "") else int(limit_value)
    if limit is not None and limit <= 0:
        raise ValueError("data.limit must be a positive integer when provided")

    backend = str(decode_reencode.get("backend", "dc_ae")).strip()
    if backend not in {"dc_ae", "jpeg"}:
        raise ValueError("decode_reencode.backend must be one of {'dc_ae', 'jpeg'}")

    jpeg_quality = int(decode_reencode.get("jpeg_quality", DEFAULT_JPEG_QUALITY))
    if jpeg_quality <= 0 or jpeg_quality > 100:
        raise ValueError("decode_reencode.jpeg_quality must be within [1, 100]")

    max_images_value = trace.get("max_images")
    max_images = None if max_images_value in (None, "") else int(max_images_value)
    if max_images is not None and max_images <= 0:
        raise ValueError("trace.max_images must be a positive integer when provided")

    enabled_signals = _validate_signal_names(parse_csv_items(signals.get("enabled")), label="signals.enabled")
    if bool(signals.get("cross_attention_instability", False)) and "cross_attention_instability" not in enabled_signals:
        enabled_signals.append("cross_attention_instability")
    if not enabled_signals:
        raise ValueError("signals.enabled must contain at least one signal")

    optional_signals = _validate_signal_names(
        parse_csv_items(signals.get("optional")),
        label="signals.optional",
    )

    entropy_bins = int(config.get("heuristics", {}).get("entropy_bins", DEFAULT_ENTROPY_BINS))
    if entropy_bins <= 1:
        raise ValueError("heuristics.entropy_bins must be greater than 1")

    return {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "records_csv": str(records_csv),
            "feature_root": str(feature_root),
        },
        "data": {
            "image_size": image_size,
            "patch_size": patch_size,
            "sample_id_column": data.get("sample_id_column"),
            "image_path_column": data.get("image_path_column"),
            "trace_path_column": data.get("trace_path_column"),
            "trace_status_column": data.get("trace_status_column"),
            "split_column": str(data.get("split_column", "split")),
            "default_split": str(data.get("default_split", "unspecified")).strip() or "unspecified",
            "split_filter": split_filter,
            "limit": limit,
        },
        "decode_reencode": {
            "backend": backend,
            "model_id": str(decode_reencode.get("model_id", DEFAULT_DC_AE_MODEL_ID)),
            "device": str(decode_reencode.get("device", "cuda")),
            "torch_dtype": str(decode_reencode.get("torch_dtype", "float32")),
            "local_files_only": bool(decode_reencode.get("local_files_only", False)),
            "jpeg_quality": jpeg_quality,
        },
        "trace": {
            "wavelet": str(trace.get("wavelet", "haar")),
            "max_images": max_images,
        },
        "signals": {
            "enabled": enabled_signals,
            "optional": optional_signals,
            "cross_attention_instability": "cross_attention_instability" in enabled_signals,
        },
        "heuristics": {
            "entropy_bins": entropy_bins,
        },
        "runtime": {
            "overwrite": bool(runtime.get("overwrite", False)),
            "allow_missing_optional_signals": bool(runtime.get("allow_missing_optional_signals", True)),
        },
    }


def load_verifier_sample_records(config: Mapping[str, Any]) -> list[VerifierSampleRecord]:
    repo_root = Path(config["paths"]["repo_root"])
    records_csv = Path(config["paths"]["records_csv"])

    with records_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if not fieldnames:
            raise VerifierFeatureDiscoveryError(f"{records_csv.name}: missing CSV header")

        data_config = config["data"]
        sample_id_column = _infer_column(
            fieldnames,
            data_config.get("sample_id_column"),
            ("sample_id", "image_id", "prompt_id"),
            required=False,
            label="sample-id column",
        )
        image_path_column = _infer_column(
            fieldnames,
            data_config.get("image_path_column"),
            ("image_path", "local_path", "source_image_path", "source_path"),
            required=True,
            label="image-path column",
        )
        trace_path_column = _infer_column(
            fieldnames,
            data_config.get("trace_path_column"),
            ("trace_path",),
            required=False,
            label="trace-path column",
        )
        trace_status_column = _infer_column(
            fieldnames,
            data_config.get("trace_status_column"),
            ("trace_status",),
            required=False,
            label="trace-status column",
        )
        split_column = _infer_column(
            fieldnames,
            data_config.get("split_column"),
            ("split",),
            required=False,
            label="split column",
        )

        split_filter = set(data_config["split_filter"])
        limit = data_config["limit"]

        records: list[VerifierSampleRecord] = []
        for row_number, row in enumerate(reader, start=2):
            raw_image_path = (row.get(image_path_column) or "").strip()
            if not raw_image_path:
                raise VerifierFeatureDiscoveryError(
                    f"{records_csv.name}: row {row_number} has empty image path in {image_path_column!r}"
                )

            image_path = _resolve_record_path(raw_image_path, repo_root)
            if not image_path.is_file():
                raise FileNotFoundError(f"{records_csv.name}: row {row_number} points to missing image {image_path}")

            if sample_id_column is not None:
                sample_id = (row.get(sample_id_column) or "").strip()
            else:
                sample_id = image_path.stem
            if not sample_id:
                raise VerifierFeatureDiscoveryError(
                    f"{records_csv.name}: row {row_number} could not derive a non-empty sample_id"
                )

            if split_column is not None:
                split = (row.get(split_column) or "").strip() or config["data"]["default_split"]
            else:
                split = config["data"]["default_split"]
            if split_filter and split not in split_filter:
                continue

            raw_trace_path = ""
            trace_path: Path | None = None
            if trace_path_column is not None:
                raw_trace_path = (row.get(trace_path_column) or "").strip()
                if raw_trace_path:
                    trace_path = _resolve_record_path(raw_trace_path, repo_root)
                    if not trace_path.is_file():
                        raise FileNotFoundError(
                            f"{records_csv.name}: row {row_number} points to missing trace file {trace_path}"
                        )

            if trace_status_column is not None:
                trace_status = (row.get(trace_status_column) or "").strip() or "unknown"
            else:
                trace_status = "present" if trace_path is not None else "missing"

            metadata = {
                key: (row.get(key) or "").strip()
                for key in ("prompt_id", "category", "stress_type", "seed")
                if key in fieldnames
            }

            records.append(
                VerifierSampleRecord(
                    sample_id=sample_id,
                    split=split,
                    image_path=image_path,
                    trace_path=trace_path,
                    trace_status=trace_status,
                    image_local_path=display_path(image_path, repo_root),
                    trace_local_path=display_path(trace_path, repo_root) if trace_path is not None else "",
                    metadata=metadata,
                )
            )
            if limit is not None and len(records) >= int(limit):
                break

    if not records:
        raise VerifierFeatureDiscoveryError("No feature records matched the current selection")
    return records


def _patchify_scalar_array(array: np.ndarray) -> np.ndarray:
    expected_shape = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    if array.shape != expected_shape:
        raise PatchMetricValidationError(f"Expected scalar image shape {expected_shape}, found {array.shape}")
    patches = array.reshape(
        DEFAULT_GRID_SIZE,
        DEFAULT_PATCH_SIZE,
        DEFAULT_GRID_SIZE,
        DEFAULT_PATCH_SIZE,
    ).transpose(0, 2, 1, 3)
    return patches.reshape(PATCH_COUNT, DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE)


def _rgb_to_luminance(image_array: np.ndarray) -> np.ndarray:
    if image_array.shape != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3):
        raise PatchMetricValidationError(
            f"Expected RGB image shape {(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3)}, found {image_array.shape}"
        )
    red = image_array[:, :, 0]
    green = image_array[:, :, 1]
    blue = image_array[:, :, 2]
    return (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)


def _compute_patchwise_mean(values: np.ndarray, *, metric_name: str) -> np.ndarray:
    patches = _patchify_scalar_array(np.asarray(values, dtype=np.float32))
    scores = np.mean(patches, axis=(1, 2), dtype=np.float64).astype(np.float32)
    return validate_patch_metric_array(scores, metric_name=metric_name)


def _compute_patchwise_mean_absolute_difference(
    first_image: Image.Image,
    second_image: Image.Image,
    *,
    metric_name: str,
) -> np.ndarray:
    first_patches = patchify_image_array(image_to_float_array(first_image))
    second_patches = patchify_image_array(image_to_float_array(second_image))
    scores = np.mean(np.abs(first_patches - second_patches), axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
    return validate_patch_metric_array(scores, metric_name=metric_name)


def _compute_decode_reencode_consistency(
    image: Image.Image,
    *,
    backend: _BaseDecodeReencodeBackend,
) -> np.ndarray:
    reconstruction = backend.reconstruct(image)
    return _compute_patchwise_mean_absolute_difference(
        image,
        reconstruction,
        metric_name="decode_reencode_consistency",
    )


def _compute_sobel_edge_magnitude(image: Image.Image) -> np.ndarray:
    luminance = _rgb_to_luminance(image_to_float_array(image))
    edge_magnitude = filters.sobel(luminance).astype(np.float32)
    return _compute_patchwise_mean(edge_magnitude, metric_name="sobel_edge_magnitude")


def _compute_local_intensity_variance(image: Image.Image) -> np.ndarray:
    luminance = _rgb_to_luminance(image_to_float_array(image))
    patches = _patchify_scalar_array(luminance)
    scores = np.var(patches, axis=(1, 2), dtype=np.float64).astype(np.float32)
    return validate_patch_metric_array(scores, metric_name="local_intensity_variance")


def _compute_patch_entropy(image: Image.Image, *, entropy_bins: int) -> np.ndarray:
    luminance = _rgb_to_luminance(image_to_float_array(image))
    patches = _patchify_scalar_array(luminance)
    scores = np.empty(PATCH_COUNT, dtype=np.float32)
    normalization = float(np.log2(entropy_bins))
    for index, patch in enumerate(patches):
        histogram, _ = np.histogram(patch, bins=entropy_bins, range=(0.0, 1.0))
        probabilities = histogram.astype(np.float64)
        probabilities /= max(1.0, float(probabilities.sum()))
        nonzero = probabilities > 0.0
        entropy = float(-np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero]), dtype=np.float64))
        scores[index] = np.float32(entropy / normalization if normalization > 0.0 else entropy)
    return validate_patch_metric_array(scores, metric_name="patch_entropy")


def _maybe_import_torch() -> Any | None:
    try:
        import torch as imported_torch
    except Exception:
        return None
    return imported_torch


def load_trace_payload(path: Path) -> dict[str, Any]:
    torch_module = _maybe_import_torch()
    if torch_module is not None:
        try:
            return dict(torch_module.load(path, map_location="cpu", weights_only=False))
        except TypeError:
            try:
                return dict(torch_module.load(path, map_location="cpu"))
            except Exception:
                pass
        except Exception:
            try:
                return dict(torch_module.load(path, map_location="cpu"))
            except Exception:
                pass

    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise SignalUnavailableError(f"{path} does not contain a mapping-like trace payload")
    return dict(payload)


def _decode_trace_images(trace_payload: Mapping[str, Any], *, max_images: int | None) -> list[Image.Image]:
    payload_format = str(trace_payload.get("format", "png_bytes"))
    if payload_format != "png_bytes":
        raise SignalUnavailableError(f"Unsupported trace payload format {payload_format!r}; expected 'png_bytes'")

    png_bytes_list = trace_payload.get("png_bytes")
    if not isinstance(png_bytes_list, Sequence) or not png_bytes_list:
        raise SignalUnavailableError("Trace payload does not contain decoded late-step PNG bytes")

    decoded_images: list[Image.Image] = []
    for payload_index, png_bytes in enumerate(png_bytes_list):
        if not isinstance(png_bytes, (bytes, bytearray)):
            raise SignalUnavailableError(f"Trace PNG payload at index {payload_index} is not bytes-like")
        with Image.open(io.BytesIO(png_bytes)) as image:
            decoded = image.convert("RGB")
        if decoded.size != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
            raise SignalUnavailableError(
                f"Trace image {payload_index} has size {decoded.size}, expected {(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)}"
            )
        decoded_images.append(decoded)

    if max_images is not None and len(decoded_images) > max_images:
        decoded_images = decoded_images[-max_images:]
    return decoded_images


def _load_trace_images_for_record(
    record: VerifierSampleRecord,
    *,
    max_images: int | None,
) -> tuple[dict[str, Any], list[Image.Image]]:
    if record.trace_path is None:
        raise SignalUnavailableError(
            f"{record.sample_id} is missing trace_path; re-run generation with --save-trace for trace-based signals"
        )
    trace_payload = load_trace_payload(record.trace_path)
    decoded_images = _decode_trace_images(trace_payload, max_images=max_images)
    return trace_payload, decoded_images


def _aggregate_pairwise_signal(
    images: Sequence[Image.Image],
    *,
    metric_name: str,
    pair_fn: Any,
) -> np.ndarray:
    if len(images) < 2:
        return validate_patch_metric_array(np.zeros(PATCH_COUNT, dtype=np.float32), metric_name=metric_name)

    pairwise_scores = [
        np.asarray(pair_fn(first_image, second_image), dtype=np.float32)
        for first_image, second_image in zip(images[:-1], images[1:])
    ]
    aggregate = np.mean(np.stack(pairwise_scores, axis=0), axis=0, dtype=np.float64).astype(np.float32)
    return validate_patch_metric_array(aggregate, metric_name=metric_name)


def _compute_late_step_update_magnitude(trace_images: Sequence[Image.Image]) -> np.ndarray:
    return _aggregate_pairwise_signal(
        trace_images,
        metric_name="late_step_update_magnitude",
        pair_fn=lambda first, second: _compute_patchwise_mean_absolute_difference(
            first,
            second,
            metric_name="late_step_update_magnitude",
        ),
    )


def _compute_wavelet_instability(trace_images: Sequence[Image.Image], *, wavelet: str) -> np.ndarray:
    def pair_fn(first_image: Image.Image, second_image: Image.Image) -> np.ndarray:
        return compute_patchwise_hf_wavelet_l1(
            patchify_image_array(image_to_float_array(first_image)),
            patchify_image_array(image_to_float_array(second_image)),
            wavelet=wavelet,
        )

    return _aggregate_pairwise_signal(
        trace_images,
        metric_name="wavelet_instability",
        pair_fn=pair_fn,
    )


def _attention_step_to_patch_vector(attention_step: Any) -> np.ndarray:
    array = np.asarray(attention_step, dtype=np.float32)
    if array.ndim == 1 and array.shape[0] == PATCH_COUNT:
        return validate_patch_metric_array(array, metric_name="cross_attention_instability")
    if array.ndim >= 2 and array.shape[-2:] == (DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE):
        flattened = np.mean(
            array.reshape(-1, DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE),
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        return validate_patch_metric_array(flattened.reshape(-1), metric_name="cross_attention_instability")
    if array.ndim == 2 and array.shape == (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
        return _compute_patchwise_mean(array, metric_name="cross_attention_instability")
    if array.ndim >= 2 and array.shape[-2:] == (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
        scalar_map = np.mean(
            array.reshape(-1, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
            axis=0,
            dtype=np.float64,
        ).astype(np.float32)
        return _compute_patchwise_mean(scalar_map, metric_name="cross_attention_instability")
    raise SignalUnavailableError(
        "Cross-attention traces must be 256-vectors, 16x16 grids, or full-resolution scalar maps"
    )


def _compute_cross_attention_instability(trace_payload: Mapping[str, Any]) -> np.ndarray:
    attention_sequence = trace_payload.get("cross_attention_maps")
    if not isinstance(attention_sequence, Sequence) or not attention_sequence:
        raise SignalUnavailableError(
            "Trace payload does not include cross_attention_maps; cross-attention instability is optional in first-pass runs"
        )

    patch_vectors = [_attention_step_to_patch_vector(step) for step in attention_sequence]
    if len(patch_vectors) < 2:
        return validate_patch_metric_array(
            np.zeros(PATCH_COUNT, dtype=np.float32),
            metric_name="cross_attention_instability",
        )
    stacked = np.stack(patch_vectors, axis=0).astype(np.float32)
    instability = np.mean(np.abs(np.diff(stacked, axis=0)), axis=0, dtype=np.float64).astype(np.float32)
    return validate_patch_metric_array(instability, metric_name="cross_attention_instability")


def build_decode_reencode_backend(config: Mapping[str, Any]) -> _BaseDecodeReencodeBackend:
    backend = str(config["decode_reencode"]["backend"])
    if backend == "jpeg":
        return JpegDecodeReencodeBackend(quality=int(config["decode_reencode"]["jpeg_quality"]))
    if backend == "dc_ae":
        return DCAEDecodeReencodeBackend(config)
    raise ValueError(f"Unsupported decode_reencode.backend {backend!r}")


def compute_signal_arrays(
    record: VerifierSampleRecord,
    *,
    config: Mapping[str, Any],
    decode_reencode_backend: _BaseDecodeReencodeBackend | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    enabled_signals = list(config["signals"]["enabled"])
    optional_signals = set(config["signals"]["optional"])
    allow_missing_optional = bool(config["runtime"]["allow_missing_optional_signals"])

    image = load_rgb_image(record.image_path)
    trace_payload: dict[str, Any] | None = None
    trace_images: list[Image.Image] | None = None

    signal_arrays: dict[str, np.ndarray] = {}
    missing_signals: dict[str, str] = {}

    for signal_name in enabled_signals:
        try:
            if signal_name == "decode_reencode_consistency":
                if decode_reencode_backend is None:
                    raise SignalUnavailableError("decode_reencode_consistency requires a decode/reencode backend")
                signal_arrays[signal_name] = _compute_decode_reencode_consistency(
                    image,
                    backend=decode_reencode_backend,
                )
            elif signal_name == "late_step_update_magnitude":
                if trace_images is None:
                    trace_payload, trace_images = _load_trace_images_for_record(
                        record,
                        max_images=config["trace"]["max_images"],
                    )
                signal_arrays[signal_name] = _compute_late_step_update_magnitude(trace_images)
            elif signal_name == "wavelet_instability":
                if trace_images is None:
                    trace_payload, trace_images = _load_trace_images_for_record(
                        record,
                        max_images=config["trace"]["max_images"],
                    )
                signal_arrays[signal_name] = _compute_wavelet_instability(
                    trace_images,
                    wavelet=str(config["trace"]["wavelet"]),
                )
            elif signal_name == "sobel_edge_magnitude":
                signal_arrays[signal_name] = _compute_sobel_edge_magnitude(image)
            elif signal_name == "local_intensity_variance":
                signal_arrays[signal_name] = _compute_local_intensity_variance(image)
            elif signal_name == "patch_entropy":
                signal_arrays[signal_name] = _compute_patch_entropy(
                    image,
                    entropy_bins=int(config["heuristics"]["entropy_bins"]),
                )
            elif signal_name == "cross_attention_instability":
                if trace_payload is None:
                    trace_payload, trace_images = _load_trace_images_for_record(
                        record,
                        max_images=config["trace"]["max_images"],
                    )
                signal_arrays[signal_name] = _compute_cross_attention_instability(trace_payload)
            else:
                raise ValueError(f"Unsupported signal {signal_name!r}")
        except SignalUnavailableError as exc:
            if allow_missing_optional and signal_name in optional_signals:
                missing_signals[signal_name] = str(exc)
                continue
            raise

    return signal_arrays, missing_signals


def feature_output_path(config: Mapping[str, Any], record: VerifierSampleRecord) -> Path:
    return Path(config["paths"]["feature_root"]) / record.split / f"{record.sample_id}{SIGNAL_SUFFIX}"


def split_index_output_path(config: Mapping[str, Any], split: str) -> Path:
    return Path(config["paths"]["feature_root"]) / split / INDEX_FILENAME


def build_feature_payload(
    record: VerifierSampleRecord,
    *,
    signal_arrays: Mapping[str, np.ndarray],
    missing_signals: Mapping[str, str],
    config: Mapping[str, Any],
    decode_reencode_backend_name: str,
) -> dict[str, np.ndarray]:
    enabled_signals = list(config["signals"]["enabled"])
    available_signals = [signal_name for signal_name in enabled_signals if signal_name in signal_arrays]
    missing_signal_names = [signal_name for signal_name in enabled_signals if signal_name in missing_signals]

    payload: dict[str, np.ndarray] = {
        "version": np.asarray(VERIFIER_FEATURE_VERSION, dtype=np.int32),
        "sample_id": np.asarray(record.sample_id),
        "split": np.asarray(record.split),
        "image_size": np.asarray(DEFAULT_IMAGE_SIZE, dtype=np.int32),
        "patch_size": np.asarray(DEFAULT_PATCH_SIZE, dtype=np.int32),
        "grid_height": np.asarray(DEFAULT_GRID_SIZE, dtype=np.int32),
        "grid_width": np.asarray(DEFAULT_GRID_SIZE, dtype=np.int32),
        "requested_signal_order": np.asarray(enabled_signals),
        "available_signal_order": np.asarray(available_signals),
        "missing_signal_order": np.asarray(missing_signal_names),
        "missing_signal_reasons_json": np.asarray(json.dumps(dict(missing_signals), sort_keys=True)),
        "image_path": np.asarray(record.image_local_path),
        "trace_path": np.asarray(record.trace_local_path),
        "trace_status": np.asarray(record.trace_status),
        "decode_reencode_backend": np.asarray(decode_reencode_backend_name),
        "wavelet": np.asarray(str(config["trace"]["wavelet"])),
    }

    for signal_name, values in signal_arrays.items():
        payload[signal_name] = validate_patch_metric_array(values, metric_name=signal_name)
    return payload


def write_feature_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _scalar_string(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        return str(value.tolist())
    return str(value)


def read_feature_payload_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        available_raw = payload["available_signal_order"] if "available_signal_order" in payload.files else np.asarray([])
        missing_raw = payload["missing_signal_order"] if "missing_signal_order" in payload.files else np.asarray([])
        missing_reasons_raw = (
            payload["missing_signal_reasons_json"]
            if "missing_signal_reasons_json" in payload.files
            else np.asarray("{}")
        )
        trace_status_raw = payload["trace_status"] if "trace_status" in payload.files else np.asarray("unknown")
        return {
            "available_signals": [str(item) for item in np.asarray(available_raw).tolist()],
            "missing_signals": [str(item) for item in np.asarray(missing_raw).tolist()],
            "missing_signal_reasons_json": _scalar_string(missing_reasons_raw),
            "trace_status": _scalar_string(trace_status_raw),
        }


def write_split_index_csv(path: Path, records: Sequence[Mapping[str, str]]) -> None:
    fieldnames = (
        "sample_id",
        "split",
        "prompt_id",
        "category",
        "stress_type",
        "seed",
        "image_path",
        "trace_path",
        "trace_status",
        "signals_path",
        "available_signals",
        "missing_signals",
        "missing_signal_reasons_json",
        "status",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(
            records,
            key=lambda item: (item.get("sample_id", ""), item.get("signals_path", "")),
        ):
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def _build_index_record(
    record: VerifierSampleRecord,
    *,
    repo_root: Path,
    signals_path: Path,
    available_signals: Sequence[str],
    missing_signals: Sequence[str],
    missing_signal_reasons_json: str,
    trace_status: str,
    status: str,
) -> dict[str, str]:
    return {
        "sample_id": record.sample_id,
        "split": record.split,
        "prompt_id": record.metadata.get("prompt_id", ""),
        "category": record.metadata.get("category", ""),
        "stress_type": record.metadata.get("stress_type", ""),
        "seed": record.metadata.get("seed", ""),
        "image_path": record.image_local_path,
        "trace_path": record.trace_local_path,
        "trace_status": trace_status,
        "signals_path": display_path(signals_path, repo_root),
        "available_signals": ",".join(available_signals),
        "missing_signals": ",".join(missing_signals),
        "missing_signal_reasons_json": missing_signal_reasons_json,
        "status": status,
    }


def extract_verifier_features(config: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    feature_root = Path(config["paths"]["feature_root"])
    feature_root.mkdir(parents=True, exist_ok=True)

    sample_records = load_verifier_sample_records(config)
    split_counts = Counter(record.split for record in sample_records)
    requested_signals = list(config["signals"]["enabled"])
    generated_feature_count = 0
    missing_optional_signal_counts: Counter[str] = Counter()
    split_index_records: dict[str, list[dict[str, str]]] = defaultdict(list)

    decode_reencode_backend: _BaseDecodeReencodeBackend | None = None
    if "decode_reencode_consistency" in requested_signals:
        decode_reencode_backend = build_decode_reencode_backend(config)
    decode_reencode_backend_name = (
        decode_reencode_backend.backend_name if decode_reencode_backend is not None else "disabled"
    )

    try:
        for record in sample_records:
            signals_path = feature_output_path(config, record)
            if signals_path.is_file() and not bool(config["runtime"]["overwrite"]):
                metadata = read_feature_payload_metadata(signals_path)
                split_index_records[record.split].append(
                    _build_index_record(
                        record,
                        repo_root=repo_root,
                        signals_path=signals_path,
                        available_signals=metadata["available_signals"],
                        missing_signals=metadata["missing_signals"],
                        missing_signal_reasons_json=metadata["missing_signal_reasons_json"],
                        trace_status=metadata["trace_status"],
                        status="skipped_existing",
                    )
                )
                for signal_name in metadata["missing_signals"]:
                    missing_optional_signal_counts[signal_name] += 1
                continue

            signal_arrays, missing_signals = compute_signal_arrays(
                record,
                config=config,
                decode_reencode_backend=decode_reencode_backend,
            )
            payload = build_feature_payload(
                record,
                signal_arrays=signal_arrays,
                missing_signals=missing_signals,
                config=config,
                decode_reencode_backend_name=decode_reencode_backend_name,
            )
            write_feature_npz(signals_path, payload)
            generated_feature_count += 1

            available_signal_names = [signal_name for signal_name in requested_signals if signal_name in signal_arrays]
            missing_signal_names = [signal_name for signal_name in requested_signals if signal_name in missing_signals]
            for signal_name in missing_signal_names:
                missing_optional_signal_counts[signal_name] += 1

            split_index_records[record.split].append(
                _build_index_record(
                    record,
                    repo_root=repo_root,
                    signals_path=signals_path,
                    available_signals=available_signal_names,
                    missing_signals=missing_signal_names,
                    missing_signal_reasons_json=json.dumps(missing_signals, sort_keys=True),
                    trace_status=record.trace_status,
                    status="generated",
                )
            )
    finally:
        if decode_reencode_backend is not None:
            decode_reencode_backend.close()

    index_paths: dict[str, str] = {}
    for split, records in split_index_records.items():
        index_path = split_index_output_path(config, split)
        write_split_index_csv(index_path, records)
        index_paths[split] = display_path(index_path, repo_root)

    return {
        "summary": {
            "records_csv": display_path(Path(config["paths"]["records_csv"]), repo_root),
            "feature_root": display_path(feature_root, repo_root),
            "requested_signals": requested_signals,
            "optional_signals": list(config["signals"]["optional"]),
            "decode_reencode_backend": decode_reencode_backend_name,
            "wavelet": str(config["trace"]["wavelet"]),
            "planned_sample_count": len(sample_records),
            "generated_feature_count": generated_feature_count,
            "split_counts": dict(split_counts),
            "missing_optional_signal_counts": dict(missing_optional_signal_counts),
            "split_index_paths": index_paths,
        },
        "split_index_records": dict(split_index_records),
    }
