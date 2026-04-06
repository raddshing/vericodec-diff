from __future__ import annotations

import csv
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pywt
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity


DEFAULT_IMAGE_SIZE = 1024
DEFAULT_PATCH_SIZE = 64
DEFAULT_GRID_SIZE = DEFAULT_IMAGE_SIZE // DEFAULT_PATCH_SIZE
PATCH_COUNT = DEFAULT_GRID_SIZE * DEFAULT_GRID_SIZE
PATCH_ERROR_VERSION = 1
PATCH_ERROR_SUFFIX = "__patch64.npz"
PATCH_PREVIEW_SUFFIX = "__patch64_preview.png"
PATCH_METRIC_NAMES = ("lpips", "one_minus_ssim", "hf_wavelet_l1")
DEFAULT_DC_AE_MODEL_ID = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"
_RESAMPLING = getattr(Image, "Resampling", Image)


class PatchMetricValidationError(ValueError):
    """Raised when patch-metric inputs or outputs violate repo constraints."""


class SampleDiscoveryError(ValueError):
    """Raised when the sample-record CSV cannot be interpreted deterministically."""


class MetricBackendUnavailableError(RuntimeError):
    """Raised when the configured LPIPS or DC-AE backend cannot be loaded."""


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    split: str
    source_path: Path
    reconstruction_path: Path
    source_local_path: str
    reconstruction_local_path: str
    metadata: dict[str, str]


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


def parse_csv_items(raw_value: str | Sequence[str] | None) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    return [str(item).strip() for item in raw_value if str(item).strip()]


def default_patch_error_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "records_csv": None,
            "reconstruction_root": "outputs/reconstructions/dc_ae",
            "error_map_root": "outputs/error_maps",
        },
        "data": {
            "image_size": DEFAULT_IMAGE_SIZE,
            "patch_size": DEFAULT_PATCH_SIZE,
            "sample_id_column": None,
            "image_path_column": None,
            "reconstruction_path_column": None,
            "split_column": "split",
            "default_split": "unspecified",
            "split_filter": [],
            "limit": None,
        },
        "runtime": {
            "overwrite": False,
        },
        "reconstruction": {
            "mode": "dc_ae",
            "model_id": DEFAULT_DC_AE_MODEL_ID,
            "device": "cuda",
            "torch_dtype": "float32",
            "local_files_only": False,
        },
        "metrics": {
            "lpips_backend": "official",
            "lpips_net": "alex",
            "lpips_batch_size": 32,
            "wavelet": "haar",
            "preview_panel_size": 256,
            "preview_percentile": 99.0,
        },
    }


def resolve_patch_error_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_patch_error_config(), raw_config)
    paths = dict(config.get("paths", {}))
    data = dict(config.get("data", {}))
    runtime = dict(config.get("runtime", {}))
    reconstruction = dict(config.get("reconstruction", {}))
    metrics = dict(config.get("metrics", {}))

    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()
    records_csv_raw = paths.get("records_csv")
    records_csv = None
    if records_csv_raw not in (None, ""):
        records_csv = resolve_path(resolved_repo_root, str(records_csv_raw)).resolve()
    reconstruction_root = resolve_path(
        resolved_repo_root,
        str(paths.get("reconstruction_root", "outputs/reconstructions/dc_ae")),
    ).resolve()
    error_map_root = resolve_path(
        resolved_repo_root,
        str(paths.get("error_map_root", "outputs/error_maps")),
    ).resolve()

    for name, path in {
        "reconstruction_root": reconstruction_root,
        "error_map_root": error_map_root,
    }.items():
        try:
            path.relative_to(resolved_repo_root)
        except ValueError as exc:
            raise ValueError(f"{name} must resolve inside repo_root for stable local paths") from exc

    image_size = int(data.get("image_size", DEFAULT_IMAGE_SIZE))
    patch_size = int(data.get("patch_size", DEFAULT_PATCH_SIZE))
    if image_size != DEFAULT_IMAGE_SIZE:
        raise ValueError(
            f"VeriCodec-Diff patch metrics are locked to {DEFAULT_IMAGE_SIZE}x{DEFAULT_IMAGE_SIZE} images"
        )
    if patch_size != DEFAULT_PATCH_SIZE:
        raise ValueError(f"VeriCodec-Diff patch metrics are locked to {DEFAULT_PATCH_SIZE}x{DEFAULT_PATCH_SIZE}")

    split_filter = parse_csv_items(data.get("split_filter"))
    limit_value = data.get("limit")
    limit = None if limit_value in (None, "") else int(limit_value)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer when provided")

    mode = str(reconstruction.get("mode", "dc_ae")).strip()
    if mode not in {"dc_ae", "precomputed"}:
        raise ValueError("reconstruction.mode must be one of {'dc_ae', 'precomputed'}")

    lpips_backend = str(metrics.get("lpips_backend", "official")).strip()
    if lpips_backend not in {"official", "pixel_l2"}:
        raise ValueError("metrics.lpips_backend must be one of {'official', 'pixel_l2'}")

    lpips_batch_size = int(metrics.get("lpips_batch_size", 32))
    if lpips_batch_size <= 0:
        raise ValueError("metrics.lpips_batch_size must be a positive integer")

    preview_panel_size = int(metrics.get("preview_panel_size", 256))
    if preview_panel_size <= 0:
        raise ValueError("metrics.preview_panel_size must be a positive integer")

    preview_percentile = float(metrics.get("preview_percentile", 99.0))
    if preview_percentile <= 0.0 or preview_percentile > 100.0:
        raise ValueError("metrics.preview_percentile must be within (0, 100]")

    return {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "records_csv": str(records_csv) if records_csv is not None else None,
            "reconstruction_root": str(reconstruction_root),
            "error_map_root": str(error_map_root),
        },
        "data": {
            "image_size": image_size,
            "patch_size": patch_size,
            "sample_id_column": data.get("sample_id_column"),
            "image_path_column": data.get("image_path_column"),
            "reconstruction_path_column": data.get("reconstruction_path_column"),
            "split_column": str(data.get("split_column", "split")),
            "default_split": str(data.get("default_split", "unspecified")).strip() or "unspecified",
            "split_filter": split_filter,
            "limit": limit,
        },
        "runtime": {
            "overwrite": bool(runtime.get("overwrite", False)),
        },
        "reconstruction": {
            "mode": mode,
            "model_id": str(reconstruction.get("model_id", DEFAULT_DC_AE_MODEL_ID)),
            "device": str(reconstruction.get("device", "cuda")),
            "torch_dtype": str(reconstruction.get("torch_dtype", "float32")),
            "local_files_only": bool(reconstruction.get("local_files_only", False)),
        },
        "metrics": {
            "lpips_backend": lpips_backend,
            "lpips_net": str(metrics.get("lpips_net", "alex")),
            "lpips_batch_size": lpips_batch_size,
            "wavelet": str(metrics.get("wavelet", "haar")),
            "preview_panel_size": preview_panel_size,
            "preview_percentile": preview_percentile,
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
            raise SampleDiscoveryError(f"Configured {label} {configured!r} is missing from the records CSV")
        return configured
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
    if required:
        raise SampleDiscoveryError(
            f"Could not infer {label}; expected one of {tuple(candidates)} in the records CSV"
        )
    return None


def _resolve_record_path(raw_path: str, repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def load_sample_records(config: Mapping[str, Any]) -> list[SampleRecord]:
    repo_root = Path(config["paths"]["repo_root"])
    records_csv_raw = config["paths"]["records_csv"]
    if records_csv_raw in (None, ""):
        raise ValueError("paths.records_csv is required")
    records_csv = Path(records_csv_raw)
    if not records_csv.is_file():
        raise FileNotFoundError(f"Sample-record CSV not found: {records_csv}")

    with records_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        if not fieldnames:
            raise SampleDiscoveryError(f"{records_csv.name}: missing CSV header")

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
            ("image_path", "local_path", "source_path"),
            required=True,
            label="source-image column",
        )
        reconstruction_path_column = _infer_column(
            fieldnames,
            data_config.get("reconstruction_path_column"),
            ("reconstruction_path", "recon_path"),
            required=False,
            label="reconstruction-path column",
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
        reconstruction_root = Path(config["paths"]["reconstruction_root"])

        records: list[SampleRecord] = []
        for row_number, row in enumerate(reader, start=2):
            raw_image_path = (row.get(image_path_column) or "").strip()
            if not raw_image_path:
                raise SampleDiscoveryError(
                    f"{records_csv.name}: row {row_number} has empty source image path in {image_path_column!r}"
                )
            source_path = _resolve_record_path(raw_image_path, repo_root)

            if not source_path.is_file():
                raise FileNotFoundError(f"{records_csv.name}: row {row_number} points to missing source image {source_path}")

            if sample_id_column is not None:
                sample_id = (row.get(sample_id_column) or "").strip()
            else:
                sample_id = Path(raw_image_path).stem
            if not sample_id:
                raise SampleDiscoveryError(f"{records_csv.name}: row {row_number} could not derive a non-empty sample_id")

            if split_column is not None:
                split = (row.get(split_column) or "").strip() or config["data"]["default_split"]
            else:
                split = config["data"]["default_split"]
            if split_filter and split not in split_filter:
                continue

            raw_reconstruction_path = ""
            if reconstruction_path_column is not None:
                raw_reconstruction_path = (row.get(reconstruction_path_column) or "").strip()
            if raw_reconstruction_path:
                reconstruction_path = _resolve_record_path(raw_reconstruction_path, repo_root)
            else:
                reconstruction_path = (reconstruction_root / split / f"{sample_id}.png").resolve()

            metadata = {
                key: (row.get(key) or "").strip()
                for key in ("category", "stress_type")
                if key in fieldnames
            }
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    split=split,
                    source_path=source_path,
                    reconstruction_path=reconstruction_path,
                    source_local_path=display_path(source_path, repo_root),
                    reconstruction_local_path=display_path(reconstruction_path, repo_root),
                    metadata=metadata,
                )
            )
            if limit is not None and len(records) >= int(limit):
                break

    if not records:
        raise SampleDiscoveryError("No sample records matched the current selection")
    return records


def load_rgb_image(path: Path, *, expected_size: int = DEFAULT_IMAGE_SIZE) -> Image.Image:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
    if rgb.size != (expected_size, expected_size):
        raise PatchMetricValidationError(
            f"{path}: expected {(expected_size, expected_size)} pixels, found {rgb.size}"
        )
    return rgb


def image_to_float_array(image: Image.Image) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.shape != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3):
        raise PatchMetricValidationError(
            f"Expected RGB image array with shape {(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3)}, found {array.shape}"
        )
    return array


def patchify_image_array(image_array: np.ndarray, *, patch_size: int = DEFAULT_PATCH_SIZE) -> np.ndarray:
    expected_shape = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3)
    if image_array.shape != expected_shape:
        raise PatchMetricValidationError(f"Expected image array shape {expected_shape}, found {image_array.shape}")
    if patch_size != DEFAULT_PATCH_SIZE:
        raise PatchMetricValidationError(f"Expected locked patch size {DEFAULT_PATCH_SIZE}, found {patch_size}")
    patches = image_array.reshape(
        DEFAULT_GRID_SIZE,
        patch_size,
        DEFAULT_GRID_SIZE,
        patch_size,
        3,
    ).transpose(0, 2, 1, 3, 4)
    return patches.reshape(PATCH_COUNT, patch_size, patch_size, 3)


def validate_patch_metric_array(values: np.ndarray, *, metric_name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size != PATCH_COUNT:
        raise PatchMetricValidationError(
            f"{metric_name} must contain {PATCH_COUNT} patch values, found {array.size}"
        )
    if not np.all(np.isfinite(array)):
        raise PatchMetricValidationError(f"{metric_name} contains non-finite values")
    if np.any(array < 0.0):
        raise PatchMetricValidationError(f"{metric_name} must stay non-negative")
    return array


class _BaseLpipsScorer:
    backend_name = "undefined"

    def score(self, source_patches: np.ndarray, reconstruction_patches: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        return None


class PixelL2LpipsScorer(_BaseLpipsScorer):
    backend_name = "pixel_l2_debug"

    def score(self, source_patches: np.ndarray, reconstruction_patches: np.ndarray) -> np.ndarray:
        squared_error = np.square(source_patches - reconstruction_patches, dtype=np.float32)
        scores = np.mean(squared_error, axis=(1, 2, 3), dtype=np.float64).astype(np.float32)
        return validate_patch_metric_array(scores, metric_name="lpips")


class OfficialLpipsScorer(_BaseLpipsScorer):
    backend_name = "lpips_official"

    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            import lpips as imported_lpips
            import torch as imported_torch
        except Exception as exc:
            raise MetricBackendUnavailableError(
                "Official LPIPS requires the local torch and lpips packages"
            ) from exc

        self.lpips = imported_lpips
        self.torch = imported_torch
        self.device = str(config["reconstruction"]["device"])
        self.batch_size = int(config["metrics"]["lpips_batch_size"])
        self.dtype = self._resolve_dtype(str(config["reconstruction"]["torch_dtype"]))
        net = str(config["metrics"]["lpips_net"])

        self.model = self.lpips.LPIPS(net=net).to(self.device)
        if self.dtype is not None:
            self.model = self.model.to(dtype=self.dtype)
        self.model.eval()

        parameters = getattr(self.model, "parameters", None)
        if callable(parameters):
            for parameter in parameters():
                if hasattr(parameter, "requires_grad_"):
                    parameter.requires_grad_(False)

    def _resolve_dtype(self, raw_dtype: str) -> Any | None:
        if raw_dtype.lower() in {"", "none", "auto"}:
            return None
        try:
            return getattr(self.torch, raw_dtype)
        except AttributeError as exc:
            raise ValueError(f"Unsupported torch dtype {raw_dtype!r}") from exc

    def score(self, source_patches: np.ndarray, reconstruction_patches: np.ndarray) -> np.ndarray:
        if source_patches.shape != reconstruction_patches.shape:
            raise PatchMetricValidationError("LPIPS source and reconstruction patch tensors must have matching shapes")

        scores = np.empty(source_patches.shape[0], dtype=np.float32)
        no_grad = getattr(self.torch, "no_grad", None)
        context = no_grad() if callable(no_grad) else None
        if context is None:
            raise MetricBackendUnavailableError("torch.no_grad is required for official LPIPS scoring")

        with context:
            for start in range(0, source_patches.shape[0], self.batch_size):
                stop = min(source_patches.shape[0], start + self.batch_size)
                source_tensor = self.torch.from_numpy(source_patches[start:stop].transpose(0, 3, 1, 2))
                reconstruction_tensor = self.torch.from_numpy(
                    reconstruction_patches[start:stop].transpose(0, 3, 1, 2)
                )
                source_tensor = (source_tensor * 2.0) - 1.0
                reconstruction_tensor = (reconstruction_tensor * 2.0) - 1.0
                to_kwargs: dict[str, Any] = {"device": self.device}
                if self.dtype is not None:
                    to_kwargs["dtype"] = self.dtype
                source_tensor = source_tensor.to(**to_kwargs)
                reconstruction_tensor = reconstruction_tensor.to(**to_kwargs)
                batch_scores = self.model(source_tensor, reconstruction_tensor)
                scores[start:stop] = (
                    batch_scores.reshape(-1).detach().float().cpu().numpy().astype(np.float32)
                )
        return validate_patch_metric_array(scores, metric_name="lpips")


def build_lpips_scorer(config: Mapping[str, Any]) -> _BaseLpipsScorer:
    backend = str(config["metrics"]["lpips_backend"])
    if backend == "pixel_l2":
        return PixelL2LpipsScorer()
    if backend == "official":
        return OfficialLpipsScorer(config)
    raise ValueError(f"Unsupported LPIPS backend {backend!r}")


class DCAEReconstructor:
    backend_name = "diffusers:AutoencoderDC"

    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            from diffusers import AutoencoderDC as ImportedAutoencoderDC
            import torch as imported_torch
        except Exception as exc:
            raise MetricBackendUnavailableError(
                "DC-AE reconstruction requires local diffusers and torch packages"
            ) from exc

        self.AutoencoderDC = ImportedAutoencoderDC
        self.torch = imported_torch
        self.device = str(config["reconstruction"]["device"])
        self.dtype = self._resolve_dtype(str(config["reconstruction"]["torch_dtype"]))
        self.model_id = str(config["reconstruction"]["model_id"])
        self.local_files_only = bool(config["reconstruction"]["local_files_only"])
        load_kwargs: dict[str, Any] = {"local_files_only": self.local_files_only}
        if self.dtype is not None:
            load_kwargs["torch_dtype"] = self.dtype
        self.model = self.AutoencoderDC.from_pretrained(self.model_id, **load_kwargs).to(self.device).eval()

    def _resolve_dtype(self, raw_dtype: str) -> Any | None:
        if raw_dtype.lower() in {"", "none", "auto"}:
            return None
        try:
            return getattr(self.torch, raw_dtype)
        except AttributeError as exc:
            raise ValueError(f"Unsupported torch dtype {raw_dtype!r}") from exc

    def reconstruct(self, image: Image.Image) -> Image.Image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = self.torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
        to_kwargs: dict[str, Any] = {"device": self.device}
        if self.dtype is not None:
            to_kwargs["dtype"] = self.dtype
        tensor = ((tensor * 2.0) - 1.0).to(**to_kwargs)

        no_grad = getattr(self.torch, "no_grad", None)
        context = no_grad() if callable(no_grad) else None
        if context is None:
            raise MetricBackendUnavailableError("torch.no_grad is required for DC-AE reconstruction")
        with context:
            encoded = self.model.encode(tensor)
            latent = getattr(encoded, "latent", encoded)
            decoded = self.model.decode(latent)
            sample = getattr(decoded, "sample", decoded)

        output = sample.detach().float().cpu().clamp(-1.0, 1.0).squeeze(0).numpy()
        output = np.transpose(output, (1, 2, 0))
        output = np.clip((output + 1.0) / 2.0, 0.0, 1.0)
        output_uint8 = np.rint(output * 255.0).astype(np.uint8)
        return Image.fromarray(output_uint8, mode="RGB")

    def close(self) -> None:
        return None


def build_reconstructor(config: Mapping[str, Any]) -> DCAEReconstructor | None:
    if config["reconstruction"]["mode"] != "dc_ae":
        return None
    return DCAEReconstructor(config)


def compute_patchwise_one_minus_ssim(
    source_patches: np.ndarray,
    reconstruction_patches: np.ndarray,
) -> np.ndarray:
    scores = np.empty(source_patches.shape[0], dtype=np.float32)
    for index in range(source_patches.shape[0]):
        ssim_value = structural_similarity(
            source_patches[index],
            reconstruction_patches[index],
            channel_axis=2,
            data_range=1.0,
        )
        scores[index] = max(0.0, float(1.0 - ssim_value))
    return validate_patch_metric_array(scores, metric_name="one_minus_ssim")


def compute_patchwise_hf_wavelet_l1(
    source_patches: np.ndarray,
    reconstruction_patches: np.ndarray,
    *,
    wavelet: str,
) -> np.ndarray:
    scores = np.empty(source_patches.shape[0], dtype=np.float32)
    for index in range(source_patches.shape[0]):
        per_channel_scores: list[float] = []
        for channel_index in range(3):
            _, source_details = pywt.dwt2(source_patches[index, :, :, channel_index], wavelet)
            _, reconstruction_details = pywt.dwt2(reconstruction_patches[index, :, :, channel_index], wavelet)
            band_scores = [
                float(np.mean(np.abs(source_band - reconstruction_band), dtype=np.float64))
                for source_band, reconstruction_band in zip(source_details, reconstruction_details)
            ]
            per_channel_scores.append(sum(band_scores) / len(band_scores))
        scores[index] = float(sum(per_channel_scores) / len(per_channel_scores))
    return validate_patch_metric_array(scores, metric_name="hf_wavelet_l1")


def compute_patch_error_arrays(
    source_image: Image.Image,
    reconstruction_image: Image.Image,
    *,
    lpips_scorer: _BaseLpipsScorer,
    wavelet: str = "haar",
) -> dict[str, np.ndarray]:
    source_patches = patchify_image_array(image_to_float_array(source_image))
    reconstruction_patches = patchify_image_array(image_to_float_array(reconstruction_image))
    lpips_values = lpips_scorer.score(source_patches, reconstruction_patches)
    one_minus_ssim = compute_patchwise_one_minus_ssim(source_patches, reconstruction_patches)
    hf_wavelet_l1 = compute_patchwise_hf_wavelet_l1(
        source_patches,
        reconstruction_patches,
        wavelet=wavelet,
    )
    return {
        "lpips": lpips_values,
        "one_minus_ssim": one_minus_ssim,
        "hf_wavelet_l1": hf_wavelet_l1,
    }


def _grid_to_heatmap_image(
    grid_values: np.ndarray,
    *,
    panel_size: int,
    percentile: float,
) -> Image.Image:
    grid = np.asarray(grid_values, dtype=np.float32).reshape(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)
    scale = float(np.percentile(grid, percentile))
    if scale <= 0.0:
        normalized = np.zeros_like(grid, dtype=np.float32)
    else:
        normalized = np.clip(grid / scale, 0.0, 1.0)
    red = np.clip(normalized * 255.0, 0.0, 255.0)
    green = np.clip(np.power(normalized, 0.7) * 220.0, 0.0, 255.0)
    blue = np.clip(np.power(normalized, 1.5) * 120.0, 0.0, 255.0)
    heatmap = np.stack([red, green, blue], axis=2).astype(np.uint8)
    return Image.fromarray(heatmap, mode="RGB").resize((panel_size, panel_size), _RESAMPLING.NEAREST)


def _thumbnail(image: Image.Image, panel_size: int) -> Image.Image:
    return image.resize((panel_size, panel_size), _RESAMPLING.BICUBIC)


def build_preview_image(
    source_image: Image.Image,
    reconstruction_image: Image.Image,
    metrics: Mapping[str, np.ndarray],
    *,
    panel_size: int,
    percentile: float,
) -> Image.Image:
    labels = ("source", "reconstruction", "lpips", "1-ssim", "hf-wavelet-l1")
    panels = [
        _thumbnail(source_image, panel_size),
        _thumbnail(reconstruction_image, panel_size),
        _grid_to_heatmap_image(metrics["lpips"], panel_size=panel_size, percentile=percentile),
        _grid_to_heatmap_image(metrics["one_minus_ssim"], panel_size=panel_size, percentile=percentile),
        _grid_to_heatmap_image(metrics["hf_wavelet_l1"], panel_size=panel_size, percentile=percentile),
    ]
    label_height = 24
    preview = Image.new("RGB", (panel_size * len(panels), panel_size + label_height), color=(24, 24, 24))
    draw = ImageDraw.Draw(preview)
    font = ImageFont.load_default()
    for index, (panel, label) in enumerate(zip(panels, labels)):
        left = index * panel_size
        preview.paste(panel, (left, label_height))
        draw.text((left + 6, 6), label, fill=(240, 240, 240), font=font)
    return preview


def build_patch_error_payload(
    record: SampleRecord,
    metrics: Mapping[str, np.ndarray],
    *,
    lpips_backend: str,
    wavelet: str,
    patch_size: int = DEFAULT_PATCH_SIZE,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> dict[str, np.ndarray]:
    return {
        "version": np.asarray(PATCH_ERROR_VERSION, dtype=np.int32),
        "sample_id": np.asarray(record.sample_id),
        "split": np.asarray(record.split),
        "patch_size": np.asarray(patch_size, dtype=np.int32),
        "image_size": np.asarray(image_size, dtype=np.int32),
        "grid_height": np.asarray(DEFAULT_GRID_SIZE, dtype=np.int32),
        "grid_width": np.asarray(DEFAULT_GRID_SIZE, dtype=np.int32),
        "metric_order": np.asarray(PATCH_METRIC_NAMES),
        "lpips_backend": np.asarray(lpips_backend),
        "wavelet": np.asarray(wavelet),
        "source_image_path": np.asarray(record.source_local_path),
        "reconstruction_image_path": np.asarray(record.reconstruction_local_path),
        "lpips": validate_patch_metric_array(metrics["lpips"], metric_name="lpips"),
        "one_minus_ssim": validate_patch_metric_array(metrics["one_minus_ssim"], metric_name="one_minus_ssim"),
        "hf_wavelet_l1": validate_patch_metric_array(metrics["hf_wavelet_l1"], metric_name="hf_wavelet_l1"),
    }


def write_patch_error_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def error_map_output_path(config: Mapping[str, Any], record: SampleRecord) -> Path:
    return Path(config["paths"]["error_map_root"]) / record.split / f"{record.sample_id}{PATCH_ERROR_SUFFIX}"


def preview_output_path(config: Mapping[str, Any], record: SampleRecord) -> Path:
    return Path(config["paths"]["error_map_root"]) / record.split / f"{record.sample_id}{PATCH_PREVIEW_SUFFIX}"


def _ensure_reconstruction(
    record: SampleRecord,
    *,
    config: Mapping[str, Any],
    reconstructor: DCAEReconstructor | None,
    source_image: Image.Image,
) -> tuple[Image.Image, str]:
    overwrite = bool(config["runtime"]["overwrite"])
    reconstruction_path = record.reconstruction_path
    if reconstruction_path.is_file() and not overwrite:
        return load_rgb_image(reconstruction_path), "existing"

    reconstruction_mode = config["reconstruction"]["mode"]
    if reconstruction_mode == "precomputed":
        raise FileNotFoundError(
            f"Precomputed reconstruction missing for {record.sample_id}: {reconstruction_path}"
        )
    if reconstructor is None:
        raise MetricBackendUnavailableError("DC-AE reconstruction was requested but no reconstructor is loaded")

    reconstruction_path.parent.mkdir(parents=True, exist_ok=True)
    reconstruction_image = reconstructor.reconstruct(source_image)
    reconstruction_image.save(reconstruction_path)
    return reconstruction_image, "generated"


def compute_patch_errors(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    error_map_root = Path(config["paths"]["error_map_root"])
    error_map_root.mkdir(parents=True, exist_ok=True)

    sample_records = load_sample_records(config)
    lpips_scorer = build_lpips_scorer(config)
    reconstructor = build_reconstructor(config)

    generated_error_maps = 0
    generated_reconstructions = 0
    split_counts = Counter(record.split for record in sample_records)
    output_records: list[dict[str, str]] = []

    try:
        for record in sample_records:
            npz_path = error_map_output_path(config, record)
            preview_path = preview_output_path(config, record)
            if npz_path.is_file() and preview_path.is_file() and not bool(config["runtime"]["overwrite"]):
                output_records.append(
                    {
                        "sample_id": record.sample_id,
                        "split": record.split,
                        "source_image_path": record.source_local_path,
                        "reconstruction_image_path": record.reconstruction_local_path,
                        "error_map_path": display_path(npz_path, repo_root),
                        "preview_path": display_path(preview_path, repo_root),
                        "reconstruction_status": "existing",
                        "status": "skipped_existing",
                    }
                )
                continue

            source_image = load_rgb_image(record.source_path)
            reconstruction_image, reconstruction_status = _ensure_reconstruction(
                record,
                config=config,
                reconstructor=reconstructor,
                source_image=source_image,
            )
            if reconstruction_status == "generated":
                generated_reconstructions += 1

            metric_arrays = compute_patch_error_arrays(
                source_image,
                reconstruction_image,
                lpips_scorer=lpips_scorer,
                wavelet=str(config["metrics"]["wavelet"]),
            )
            payload = build_patch_error_payload(
                record,
                metric_arrays,
                lpips_backend=lpips_scorer.backend_name,
                wavelet=str(config["metrics"]["wavelet"]),
            )
            write_patch_error_npz(npz_path, payload)
            preview = build_preview_image(
                source_image,
                reconstruction_image,
                metric_arrays,
                panel_size=int(config["metrics"]["preview_panel_size"]),
                percentile=float(config["metrics"]["preview_percentile"]),
            )
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            preview.save(preview_path)

            generated_error_maps += 1
            output_records.append(
                {
                    "sample_id": record.sample_id,
                    "split": record.split,
                    "source_image_path": record.source_local_path,
                    "reconstruction_image_path": display_path(record.reconstruction_path, repo_root),
                    "error_map_path": display_path(npz_path, repo_root),
                    "preview_path": display_path(preview_path, repo_root),
                    "reconstruction_status": reconstruction_status,
                    "status": "generated",
                }
            )
    finally:
        lpips_scorer.close()
        if reconstructor is not None:
            reconstructor.close()

    return {
        "summary": {
            "records_csv": display_path(Path(config["paths"]["records_csv"]), repo_root),
            "error_map_root": display_path(error_map_root, repo_root),
            "reconstruction_root": display_path(Path(config["paths"]["reconstruction_root"]), repo_root),
            "reconstruction_mode": config["reconstruction"]["mode"],
            "lpips_backend": lpips_scorer.backend_name,
            "wavelet": config["metrics"]["wavelet"],
            "patch_size": int(config["data"]["patch_size"]),
            "image_size": int(config["data"]["image_size"]),
            "planned_sample_count": len(sample_records),
            "generated_error_map_count": generated_error_maps,
            "generated_reconstruction_count": generated_reconstructions,
            "split_counts": dict(split_counts),
        },
        "records": output_records,
    }


def write_patch_error_records_csv(path: Path, records: Sequence[Mapping[str, str]]) -> None:
    fieldnames = (
        "sample_id",
        "split",
        "source_image_path",
        "reconstruction_image_path",
        "error_map_path",
        "preview_path",
        "reconstruction_status",
        "status",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})
