from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw


GB = 1024**3
DEFAULT_IMAGE_SIZE = 1024
DEFAULT_NUM_REFERENCE_TOKENS = 64
DEFAULT_REFERENCE_EMBED_DIM = 2048
DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "Visualize a scene that closely resembles the provided images, "
    "capturing the essence and details described in this prompt:\n"
)
DEFAULT_NEGATIVE_PROMPT = (
    "A collage of images, monochrome, lowres, bad anatomy, worst quality, low quality"
)
PROFILE_CSV_FIELDS = (
    "prompt_id",
    "seed",
    "ref_count",
    "ref_tokens_cond_bytes",
    "ref_tokens_uncond_bytes",
    "persistent_ref_interface_bytes",
    "persistent_ref_interface_gb",
    "persistent_ref_interface_share",
    "reference_encode_peak_delta_gb",
    "reference_encode_peak_vram_gb",
    "total_peak_vram_gb",
)
RUN_CSV_FIELDS = (
    "sample_id",
    "prompt_id",
    "seed",
    "ref_count",
    "image_path",
    "generation_seconds",
    "status",
)


class EasyRefRunnerError(RuntimeError):
    """Raised when the EasyRef wrapper cannot satisfy a requested run."""


class EasyRefBackendUnavailableError(EasyRefRunnerError):
    """Raised when the configured EasyRef backend cannot be loaded."""


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


def _validate_suite_name(raw_suite: str) -> str:
    suite = raw_suite.strip()
    if not suite:
        raise ValueError("suite must be a non-empty path token")
    suite_path = Path(suite)
    if len(suite_path.parts) != 1 or suite_path.name != suite or suite in {".", ".."}:
        raise ValueError("suite must be a single path component")
    return suite


def _parse_int_sequence(raw_value: str | Sequence[int] | Sequence[str] | None, *, label: str) -> list[int]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        items = [item.strip() for item in raw_value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in raw_value if str(item).strip()]
    values: list[int] = []
    for item in items:
        value = int(item)
        if value <= 0:
            raise ValueError(f"{label} entries must be positive integers")
        values.append(value)
    return values


def _ensure_inside_repo(name: str, path: Path, repo_root: Path) -> None:
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{name} must resolve inside repo_root for stable local paths") from exc


def _normalize_model_locator(repo_root: Path, raw_value: str | None) -> str:
    locator = str(raw_value or "").strip()
    if not locator:
        return ""
    direct_path = Path(locator).expanduser()
    if direct_path.exists():
        return str(direct_path.resolve())
    repo_relative_path = (repo_root / direct_path).resolve()
    if repo_relative_path.exists():
        return str(repo_relative_path)
    return locator


def default_easyref_config() -> dict[str, Any]:
    repeated_aragaki = [
        "baselines/external/templex98_easyref/assets/aragaki_identity/1.jpg",
        "baselines/external/templex98_easyref/assets/aragaki_identity/2.webp",
        "baselines/external/templex98_easyref/assets/aragaki_identity/3.webp",
        "baselines/external/templex98_easyref/assets/aragaki_identity/4.jpeg",
        "baselines/external/templex98_easyref/assets/aragaki_identity/5.webp",
        "baselines/external/templex98_easyref/assets/aragaki_identity/1.jpg",
        "baselines/external/templex98_easyref/assets/aragaki_identity/2.webp",
        "baselines/external/templex98_easyref/assets/aragaki_identity/3.webp",
    ]
    return {
        "paths": {
            "repo_root": ".",
            "baseline_root": "baselines/external/templex98_easyref",
            "output_root": "outputs/baselines/easyref",
            "profile_output_dir": "outputs/metrics/turbocontext_gate",
            "suite": "turbocontext_easyref_tiny",
        },
        "backend": {
            "kind": "easyref",
            "base_model_path": "stabilityai/stable-diffusion-xl-base-1.0",
            "multimodal_llm_path": "Qwen/Qwen2-VL-2B-Instruct",
            "easyref_checkpoint_path": "checkpoints/EasyRef/model.safetensors",
            "easyref_repo_id": "zongzhuofan/EasyRef",
            "easyref_checkpoint_filename": "model.safetensors",
            "device": "cuda",
            "torch_dtype": "float16",
            "num_tokens": DEFAULT_NUM_REFERENCE_TOKENS,
            "use_lora": True,
            "lora_rank": 128,
            "cond_image_size": 336,
            "local_files_only": False,
        },
        "generation": {
            "height": DEFAULT_IMAGE_SIZE,
            "width": DEFAULT_IMAGE_SIZE,
            "num_samples": 1,
            "num_inference_steps": 30,
            "scale": 1.0,
            "ref_count": 4,
            "overwrite": False,
            "system_prompt_template": DEFAULT_SYSTEM_PROMPT_TEMPLATE,
            "unconditional_system_prompt": DEFAULT_SYSTEM_PROMPT_TEMPLATE,
        },
        "profiling": {
            "ref_counts": [1, 2, 4, 8],
            "verify_ip_attention": False,
        },
        "suite": {
            "items": [
                {
                    "prompt_id": "easyref_smoke_aragaki_001",
                    "prompt": (
                        "A cinematic studio portrait of the same woman, natural window light, "
                        "shallow depth of field."
                    ),
                    "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
                    "seed": 2026040801,
                    "reference_images": repeated_aragaki,
                }
            ]
        },
    }


def build_run_output_dir(config: Mapping[str, Any], ref_count: int) -> Path:
    output_root = Path(config["paths"]["output_root"])
    suite = str(config["paths"]["suite"])
    return (output_root / suite / f"ref_count_{int(ref_count)}").resolve()


def resolve_easyref_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_easyref_config(), raw_config)
    paths = dict(config.get("paths", {}))
    backend = dict(config.get("backend", {}))
    generation = dict(config.get("generation", {}))
    profiling = dict(config.get("profiling", {}))
    suite = dict(config.get("suite", {}))

    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()
    baseline_root = resolve_path(
        resolved_repo_root,
        str(paths.get("baseline_root", "baselines/external/templex98_easyref")),
    ).resolve()
    output_root = resolve_path(
        resolved_repo_root,
        str(paths.get("output_root", "outputs/baselines/easyref")),
    ).resolve()
    profile_output_dir = resolve_path(
        resolved_repo_root,
        str(paths.get("profile_output_dir", "outputs/metrics/turbocontext_gate")),
    ).resolve()
    suite_name = _validate_suite_name(str(paths.get("suite", "turbocontext_easyref_tiny")))

    _ensure_inside_repo("baseline_root", baseline_root, resolved_repo_root)
    _ensure_inside_repo("output_root", output_root, resolved_repo_root)
    _ensure_inside_repo("profile_output_dir", profile_output_dir, resolved_repo_root)

    backend_kind = str(backend.get("kind", "easyref")).strip().lower()
    if backend_kind not in {"easyref", "mock"}:
        raise ValueError("backend.kind must be one of {'easyref', 'mock'}")

    height = int(generation.get("height", DEFAULT_IMAGE_SIZE))
    width = int(generation.get("width", DEFAULT_IMAGE_SIZE))
    if (height, width) != (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE):
        raise ValueError(f"EasyRef runs are locked to {DEFAULT_IMAGE_SIZE}x{DEFAULT_IMAGE_SIZE}")

    num_samples = int(generation.get("num_samples", 1))
    if num_samples <= 0:
        raise ValueError("generation.num_samples must be a positive integer")

    num_inference_steps = int(generation.get("num_inference_steps", 30))
    if num_inference_steps <= 0:
        raise ValueError("generation.num_inference_steps must be a positive integer")

    ref_count = int(generation.get("ref_count", 4))
    if ref_count <= 0:
        raise ValueError("generation.ref_count must be a positive integer")

    ref_counts = _parse_int_sequence(
        profiling.get("ref_counts", [1, 2, 4, 8]),
        label="profiling.ref_counts",
    )
    if not ref_counts:
        raise ValueError("profiling.ref_counts must provide at least one reference count")

    system_prompt_template = str(
        generation.get("system_prompt_template", DEFAULT_SYSTEM_PROMPT_TEMPLATE)
    )
    unconditional_system_prompt = str(
        generation.get("unconditional_system_prompt", system_prompt_template)
    )
    if not system_prompt_template.strip():
        raise ValueError("generation.system_prompt_template must be non-empty")
    if not unconditional_system_prompt.strip():
        raise ValueError("generation.unconditional_system_prompt must be non-empty")

    items = list(suite.get("items") or [])
    if not items:
        raise ValueError("suite.items must contain at least one prompt item")

    required_ref_count = max(ref_count, max(ref_counts))
    resolved_items: list[dict[str, Any]] = []
    seen_prompt_ids: set[str] = set()
    for index, raw_item in enumerate(items, start=1):
        if not isinstance(raw_item, Mapping):
            raise ValueError(f"suite.items[{index - 1}] must be a mapping")

        prompt_id = str(raw_item.get("prompt_id", "")).strip()
        prompt = str(raw_item.get("prompt", "")).strip()
        negative_prompt = str(raw_item.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)).strip()
        seed = int(raw_item.get("seed", 0))
        raw_references = list(raw_item.get("reference_images") or [])
        if not prompt_id:
            raise ValueError(f"suite.items[{index - 1}] must define prompt_id")
        if "/" in prompt_id or prompt_id in {".", ".."}:
            raise ValueError(f"suite.items[{index - 1}] prompt_id must be a stable path token")
        if prompt_id in seen_prompt_ids:
            raise ValueError(f"suite.items contains duplicate prompt_id {prompt_id!r}")
        if not prompt:
            raise ValueError(f"suite.items[{index - 1}] must define prompt")
        if seed <= 0:
            raise ValueError(f"suite.items[{index - 1}] must use a positive integer seed")
        if len(raw_references) < required_ref_count:
            raise ValueError(
                f"suite.items[{index - 1}] only provides {len(raw_references)} references; "
                f"{required_ref_count} are required for the configured sweep"
            )

        resolved_reference_paths: list[str] = []
        for raw_reference in raw_references:
            resolved_reference = resolve_path(resolved_repo_root, str(raw_reference)).resolve()
            if not resolved_reference.is_file():
                raise FileNotFoundError(f"Reference image not found: {resolved_reference}")
            resolved_reference_paths.append(str(resolved_reference))

        seen_prompt_ids.add(prompt_id)
        resolved_items.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "negative_prompt": negative_prompt or DEFAULT_NEGATIVE_PROMPT,
                "seed": seed,
                "reference_images": resolved_reference_paths,
            }
        )

    resolved_config = {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "baseline_root": str(baseline_root),
            "output_root": str(output_root),
            "output_dir": str((output_root / suite_name / f"ref_count_{ref_count}").resolve()),
            "profile_output_dir": str(profile_output_dir),
            "suite": suite_name,
        },
        "backend": {
            "kind": backend_kind,
            "base_model_path": _normalize_model_locator(
                resolved_repo_root, backend.get("base_model_path")
            ),
            "multimodal_llm_path": _normalize_model_locator(
                resolved_repo_root, backend.get("multimodal_llm_path")
            ),
            "easyref_checkpoint_path": _normalize_model_locator(
                resolved_repo_root, backend.get("easyref_checkpoint_path")
            ),
            "easyref_repo_id": str(backend.get("easyref_repo_id", "zongzhuofan/EasyRef")).strip(),
            "easyref_checkpoint_filename": str(
                backend.get("easyref_checkpoint_filename", "model.safetensors")
            ).strip(),
            "device": str(backend.get("device", "cuda")).strip(),
            "torch_dtype": str(backend.get("torch_dtype", "float16")).strip(),
            "num_tokens": int(backend.get("num_tokens", DEFAULT_NUM_REFERENCE_TOKENS)),
            "use_lora": bool(backend.get("use_lora", True)),
            "lora_rank": int(backend.get("lora_rank", 128)),
            "cond_image_size": int(backend.get("cond_image_size", 336)),
            "local_files_only": bool(backend.get("local_files_only", False)),
        },
        "generation": {
            "height": height,
            "width": width,
            "num_samples": num_samples,
            "num_inference_steps": num_inference_steps,
            "scale": float(generation.get("scale", 1.0)),
            "ref_count": ref_count,
            "overwrite": bool(generation.get("overwrite", False)),
            "system_prompt_template": system_prompt_template,
            "unconditional_system_prompt": unconditional_system_prompt,
        },
        "profiling": {
            "ref_counts": ref_counts,
            "verify_ip_attention": bool(profiling.get("verify_ip_attention", False)),
        },
        "suite": {
            "items": resolved_items,
        },
    }
    return resolved_config


@dataclass(frozen=True)
class EasyRefRunSpec:
    sample_id: str
    prompt_id: str
    prompt: str
    negative_prompt: str
    seed: int
    ref_count: int
    reference_images: tuple[Path, ...]
    image_path: Path
    image_local_path: str


def build_sample_id(prompt_id: str, seed: int) -> str:
    return f"{prompt_id}__seed_{int(seed)}"


def build_easyref_run_plan(config: Mapping[str, Any], *, ref_count: int | None = None) -> list[EasyRefRunSpec]:
    repo_root = Path(config["paths"]["repo_root"])
    effective_ref_count = int(ref_count if ref_count is not None else config["generation"]["ref_count"])
    output_dir = build_run_output_dir(config, effective_ref_count)

    plan: list[EasyRefRunSpec] = []
    for item in config["suite"]["items"]:
        sample_id = build_sample_id(item["prompt_id"], int(item["seed"]))
        image_path = output_dir / f"{sample_id}.png"
        references = tuple(Path(path) for path in item["reference_images"][:effective_ref_count])
        plan.append(
            EasyRefRunSpec(
                sample_id=sample_id,
                prompt_id=item["prompt_id"],
                prompt=item["prompt"],
                negative_prompt=item["negative_prompt"],
                seed=int(item["seed"]),
                ref_count=effective_ref_count,
                reference_images=references,
                image_path=image_path,
                image_local_path=display_path(image_path, repo_root),
            )
        )
    return plan


def write_run_records_csv(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in RUN_CSV_FIELDS})


def write_profile_records_csv(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field, "") for field in PROFILE_CSV_FIELDS}
            for field in (
                "persistent_ref_interface_gb",
                "persistent_ref_interface_share",
                "reference_encode_peak_delta_gb",
                "reference_encode_peak_vram_gb",
                "total_peak_vram_gb",
            ):
                value = row.get(field, "")
                if isinstance(value, (int, float)):
                    row[field] = f"{float(value):.6f}"
            writer.writerow(row)


def build_reference_interface_metrics(
    *,
    ref_tokens_cond_bytes: int,
    ref_tokens_uncond_bytes: int,
    total_peak_vram_gb: float,
    cond_reference_peak: Mapping[str, Any],
    uncond_reference_peak: Mapping[str, Any],
) -> dict[str, int | float]:
    persistent_ref_interface_bytes = int(ref_tokens_cond_bytes) + int(ref_tokens_uncond_bytes)
    persistent_ref_interface_gb = round(float(persistent_ref_interface_bytes) / GB, 6)
    persistent_ref_interface_share = (
        round(persistent_ref_interface_gb / float(total_peak_vram_gb), 6) if total_peak_vram_gb else 0.0
    )
    reference_encode_peak_delta_gb = round(
        max(
            float(cond_reference_peak["peak_delta_vram_gb"]),
            float(uncond_reference_peak["peak_delta_vram_gb"]),
        ),
        6,
    )
    reference_encode_peak_vram_gb = round(
        max(
            float(cond_reference_peak["peak_vram_gb"]),
            float(uncond_reference_peak["peak_vram_gb"]),
        ),
        6,
    )
    return {
        "ref_tokens_cond_bytes": int(ref_tokens_cond_bytes),
        "ref_tokens_uncond_bytes": int(ref_tokens_uncond_bytes),
        "persistent_ref_interface_bytes": persistent_ref_interface_bytes,
        "persistent_ref_interface_gb": persistent_ref_interface_gb,
        "persistent_ref_interface_share": persistent_ref_interface_share,
        "reference_encode_peak_delta_gb": reference_encode_peak_delta_gb,
        "reference_encode_peak_vram_gb": reference_encode_peak_vram_gb,
    }


def _coerce_first_image(payload: Any) -> Image.Image:
    if isinstance(payload, Image.Image):
        return payload
    if isinstance(payload, (list, tuple)) and payload:
        first = payload[0]
        if isinstance(first, Image.Image):
            return first
    raise EasyRefRunnerError(f"Expected a PIL.Image.Image, received {type(payload).__name__}")


class MockEasyRefBackend:
    backend_name = "mock:easyref"
    measurement_mode = "mock_direct_tensor_bytes"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.generation_config = dict(config["generation"])
        self.profiling_config = dict(config["profiling"])
        self.backend_load_resident_gb = 1.75

    def _make_mock_image(self, spec: EasyRefRunSpec) -> Image.Image:
        digest = hashlib.sha256(
            f"{spec.prompt_id}:{spec.seed}:{spec.ref_count}".encode("utf-8")
        ).digest()
        color = (digest[0], digest[1], digest[2])
        accent = (digest[3], digest[4], digest[5])
        image = Image.new(
            "RGB",
            (int(self.generation_config["width"]), int(self.generation_config["height"])),
            color=color,
        )
        drawer = ImageDraw.Draw(image)
        drawer.rectangle((64, 64, 960, 960), outline=accent, width=24)
        drawer.rectangle((128, 128, 896, 896), outline=color[::-1], width=16)
        return image

    def generate_image(self, spec: EasyRefRunSpec) -> Image.Image:
        return self._make_mock_image(spec)

    def profile_generation(self, spec: EasyRefRunSpec) -> dict[str, Any]:
        ref_count = int(spec.ref_count)
        component_peaks_gb = {
            "cond_reference_encode": {
                "start_vram_gb": 1.75,
                "end_vram_gb": round(1.85 + 0.02 * ref_count, 6),
                "peak_vram_gb": round(2.10 + 0.07 * ref_count, 6),
                "peak_delta_vram_gb": round(0.35 + 0.07 * ref_count, 6),
            },
            "uncond_reference_encode": {
                "start_vram_gb": 1.85,
                "end_vram_gb": 1.88,
                "peak_vram_gb": round(2.02 + 0.01 * ref_count, 6),
                "peak_delta_vram_gb": round(0.17 + 0.01 * ref_count, 6),
            },
            "prompt_conditioning": {
                "start_vram_gb": 1.88,
                "end_vram_gb": round(1.93 + 0.04 * ref_count, 6),
                "peak_vram_gb": round(2.25 + 0.09 * ref_count, 6),
                "peak_delta_vram_gb": round(0.37 + 0.09 * ref_count, 6),
            },
            "diffusion_decode": {
                "start_vram_gb": round(1.93 + 0.04 * ref_count, 6),
                "end_vram_gb": 1.96,
                "peak_vram_gb": round(2.95 + 0.13 * ref_count, 6),
                "peak_delta_vram_gb": round(1.02 + 0.09 * ref_count, 6),
            },
        }
        total_peak_vram_gb = round(
            max(component["peak_vram_gb"] for component in component_peaks_gb.values()),
            6,
        )
        per_interface_tensor_bytes = (
            int(self.generation_config["num_samples"])
            * DEFAULT_NUM_REFERENCE_TOKENS
            * DEFAULT_REFERENCE_EMBED_DIM
            * 2
        )
        reference_interface_metrics = build_reference_interface_metrics(
            ref_tokens_cond_bytes=per_interface_tensor_bytes,
            ref_tokens_uncond_bytes=per_interface_tensor_bytes,
            total_peak_vram_gb=total_peak_vram_gb,
            cond_reference_peak=component_peaks_gb["cond_reference_encode"],
            uncond_reference_peak=component_peaks_gb["uncond_reference_encode"],
        )
        return {
            "measurement_mode": self.measurement_mode,
            "backend_load_resident_gb": self.backend_load_resident_gb,
            "component_peaks_gb": component_peaks_gb,
            "total_peak_vram_gb": total_peak_vram_gb,
            **reference_interface_metrics,
        }

    def close(self) -> None:
        return None


class CudaMemoryTracker:
    def __init__(self, torch_module: Any, device: str) -> None:
        self.torch = torch_module
        self.device = device

    def is_available(self) -> bool:
        cuda = getattr(self.torch, "cuda", None)
        return bool(cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available())

    def synchronize(self) -> None:
        self.torch.cuda.synchronize(self.device)

    def current_gb(self) -> float:
        return float(self.torch.cuda.memory_allocated(self.device)) / GB

    def reset_peak(self) -> None:
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak_gb(self) -> float:
        return float(self.torch.cuda.max_memory_allocated(self.device)) / GB

    def measure_span(self, fn: Any) -> tuple[Any, dict[str, float]]:
        self.synchronize()
        start_gb = self.current_gb()
        self.reset_peak()
        result = fn()
        self.synchronize()
        end_gb = self.current_gb()
        peak_gb = self.peak_gb()
        return result, {
            "start_vram_gb": round(start_gb, 6),
            "end_vram_gb": round(end_gb, 6),
            "peak_vram_gb": round(peak_gb, 6),
            "peak_delta_vram_gb": round(max(0.0, peak_gb - start_gb), 6),
        }


class EasyRefDiffusersBackend:
    backend_name = "easyref:sdxl"
    measurement_mode = "direct_tensor_bytes"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.backend_config = dict(config["backend"])
        self.generation_config = dict(config["generation"])
        self.profiling_config = dict(config["profiling"])
        self.repo_root = Path(config["paths"]["repo_root"])
        self.baseline_root = Path(config["paths"]["baseline_root"])
        self.torch = self._load_torch()
        self.pipeline_cls = self._load_pipeline_cls()
        self.easyref_cls, self.generator_factory = self._load_easyref_support()
        self.pipe = self._load_pipeline()
        self.easyref = self._load_easyref()
        self.ip_attention_verification_records: list[dict[str, Any]] = []
        self._ip_attention_verification_enabled = bool(
            self.profiling_config.get("verify_ip_attention", False)
        )
        self._ip_attention_verification_processor_count = 0
        if self._ip_attention_verification_enabled:
            self._install_ip_attention_verification_hooks()
        self.memory_tracker = CudaMemoryTracker(self.torch, self.backend_config["device"])
        if not self.memory_tracker.is_available():
            raise EasyRefBackendUnavailableError("CUDA is required for EasyRef VRAM profiling")
        self.memory_tracker.synchronize()
        self.backend_load_resident_gb = round(self.memory_tracker.current_gb(), 6)

    def _load_torch(self) -> Any:
        try:
            import torch as imported_torch
        except Exception as exc:
            raise EasyRefBackendUnavailableError("torch is required for the EasyRef backend") from exc
        return imported_torch

    def _load_pipeline_cls(self) -> type[Any]:
        try:
            from diffusers import StableDiffusionXLPipeline as imported_pipeline
        except Exception as exc:
            raise EasyRefBackendUnavailableError(
                "diffusers.StableDiffusionXLPipeline is required for the EasyRef backend"
            ) from exc
        return imported_pipeline

    def _load_easyref_support(self) -> tuple[type[Any], Any]:
        if not self.baseline_root.is_dir():
            raise EasyRefBackendUnavailableError(f"EasyRef checkout not found at {self.baseline_root}")
        if str(self.baseline_root) not in sys.path:
            sys.path.insert(0, str(self.baseline_root))
        try:
            from ip_adapter import EasyRef as imported_easyref
            from ip_adapter.utils import get_generator as imported_get_generator
        except Exception as exc:
            raise EasyRefBackendUnavailableError(
                "The pinned EasyRef checkout could not be imported"
            ) from exc
        return imported_easyref, imported_get_generator

    def _resolve_dtype(self, raw_dtype: str | None) -> Any | None:
        if raw_dtype in (None, "", "auto", "none", "None"):
            return None
        try:
            return getattr(self.torch, str(raw_dtype))
        except AttributeError as exc:
            raise ValueError(f"Unsupported torch dtype {raw_dtype!r}") from exc

    def _resolve_checkpoint_path(self) -> str:
        checkpoint_path = str(self.backend_config.get("easyref_checkpoint_path", "")).strip()
        if checkpoint_path:
            direct_path = Path(checkpoint_path).expanduser()
            if direct_path.is_file():
                return str(direct_path.resolve())
            repo_relative_path = (self.repo_root / direct_path).resolve()
            if repo_relative_path.is_file():
                return str(repo_relative_path)

        repo_id = str(self.backend_config.get("easyref_repo_id", "")).strip()
        filename = str(self.backend_config.get("easyref_checkpoint_filename", "")).strip()
        if not repo_id or not filename:
            raise EasyRefBackendUnavailableError(
                "backend.easyref_checkpoint_path must point to a file, or easyref_repo_id/"
                "easyref_checkpoint_filename must be set"
            )
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            raise EasyRefBackendUnavailableError(
                "huggingface_hub is required when EasyRef checkpoints are resolved from a repo id"
            ) from exc
        try:
            return str(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_files_only=bool(self.backend_config["local_files_only"]),
                )
            )
        except Exception as exc:
            raise EasyRefBackendUnavailableError(
                f"Could not resolve EasyRef checkpoint {repo_id}/{filename}"
            ) from exc

    def _load_pipeline(self) -> Any:
        torch_dtype = self._resolve_dtype(self.backend_config.get("torch_dtype"))
        load_kwargs: dict[str, Any] = {
            "add_watermarker": False,
            "local_files_only": bool(self.backend_config["local_files_only"]),
        }
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        pipeline = self.pipeline_cls.from_pretrained(
            self.backend_config["base_model_path"],
            **load_kwargs,
        )
        pipeline.to(self.backend_config["device"])
        return pipeline

    def _load_easyref(self) -> Any:
        return self.easyref_cls(
            self.pipe,
            self.backend_config["multimodal_llm_path"],
            self._resolve_checkpoint_path(),
            self.backend_config["device"],
            num_tokens=int(self.backend_config["num_tokens"]),
            use_lora=bool(self.backend_config["use_lora"]),
            lora_rank=int(self.backend_config["lora_rank"]),
            cond_image_size=int(self.backend_config["cond_image_size"]),
        )

    def _load_reference_images(self, reference_images: Sequence[Path]) -> list[Image.Image]:
        images: list[Image.Image] = []
        for path in reference_images:
            with Image.open(path) as handle:
                images.append(handle.convert("RGB"))
        return images

    def _system_prompts(self, prompt: str) -> list[str]:
        template = str(self.generation_config["system_prompt_template"])
        unconditional = str(self.generation_config["unconditional_system_prompt"])
        return [f"{template}{prompt}", unconditional]

    def _tensor_nbytes(self, tensor: Any) -> int:
        numel = getattr(tensor, "numel", None)
        element_size = getattr(tensor, "element_size", None)
        if callable(numel) and callable(element_size):
            return int(numel()) * int(element_size())
        raise EasyRefRunnerError(f"Could not measure tensor bytes for {type(tensor).__name__}")

    def _install_ip_attention_verification_hooks(self) -> None:
        processors = getattr(getattr(self.easyref.pipe, "unet", None), "attn_processors", {})
        processor_count = 0

        for processor_name, processor in getattr(processors, "items", lambda: [])():
            if processor.__class__.__name__ not in {"IPAttnProcessor2_0", "LoRAIPAttnProcessor2_0"}:
                continue

            processor_count += 1

            def _record(payload: Mapping[str, Any], *, name: str = str(processor_name)) -> None:
                if any(record.get("processor_name") == name for record in self.ip_attention_verification_records):
                    return
                payload_record = dict(payload)
                payload_record["processor_name"] = name
                self.ip_attention_verification_records.append(payload_record)

            setattr(processor, "ip_attention_verification_hook", _record)

        self._ip_attention_verification_processor_count = processor_count

    def _cleanup_after_run(self) -> None:
        gc.collect()
        if hasattr(self.torch.cuda, "empty_cache"):
            self.torch.cuda.empty_cache()
        self.memory_tracker.synchronize()

    def generate_image(self, spec: EasyRefRunSpec) -> Image.Image:
        reference_images = self._load_reference_images(spec.reference_images)
        results = self.easyref.generate(
            pil_image=reference_images,
            system_prompt=self._system_prompts(spec.prompt),
            prompt=spec.prompt,
            negative_prompt=spec.negative_prompt,
            scale=float(self.generation_config["scale"]),
            num_samples=int(self.generation_config["num_samples"]),
            seed=int(spec.seed),
            num_inference_steps=int(self.generation_config["num_inference_steps"]),
            height=int(self.generation_config["height"]),
            width=int(self.generation_config["width"]),
        )
        image = _coerce_first_image(results)
        self._cleanup_after_run()
        return image

    def profile_generation(self, spec: EasyRefRunSpec) -> dict[str, Any]:
        reference_images = self._load_reference_images(spec.reference_images)
        system_prompts = self._system_prompts(spec.prompt)
        negative_prompt = spec.negative_prompt or DEFAULT_NEGATIVE_PROMPT
        self.easyref.set_scale(float(self.generation_config["scale"]))
        self.ip_attention_verification_records = []

        image_prompt_embeds, cond_reference_peak = self.memory_tracker.measure_span(
            lambda: self.easyref.get_image_embeds(reference_images, system_prompts[0])
        )
        uncond_image_prompt_embeds, uncond_reference_peak = self.memory_tracker.measure_span(
            lambda: self.easyref.get_image_embeds(
                Image.new(mode="RGB", size=(512, 512)),
                system_prompts[1],
            )
        )

        def build_conditioning() -> dict[str, Any]:
            bs_embed, seq_len, _ = image_prompt_embeds.shape
            conditioned = image_prompt_embeds.repeat(1, int(self.generation_config["num_samples"]), 1)
            conditioned = conditioned.view(bs_embed * int(self.generation_config["num_samples"]), seq_len, -1)
            unconditioned = uncond_image_prompt_embeds.repeat(1, int(self.generation_config["num_samples"]), 1)
            unconditioned = unconditioned.view(
                bs_embed * int(self.generation_config["num_samples"]),
                seq_len,
                -1,
            )
            with self.torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self.easyref.pipe.encode_prompt(
                    [spec.prompt],
                    num_images_per_prompt=int(self.generation_config["num_samples"]),
                    do_classifier_free_guidance=True,
                    negative_prompt=[negative_prompt],
                )
                prompt_embeds = self.torch.cat([prompt_embeds, conditioned], dim=1)
                negative_prompt_embeds = self.torch.cat([negative_prompt_embeds, unconditioned], dim=1)
            return {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            }

        conditioning, prompt_conditioning_peak = self.memory_tracker.measure_span(build_conditioning)
        generator = self.generator_factory(int(spec.seed), self.backend_config["device"])

        def run_pipe() -> Any:
            return self.easyref.pipe(
                prompt_embeds=conditioning["prompt_embeds"],
                negative_prompt_embeds=conditioning["negative_prompt_embeds"],
                pooled_prompt_embeds=conditioning["pooled_prompt_embeds"],
                negative_pooled_prompt_embeds=conditioning["negative_pooled_prompt_embeds"],
                num_inference_steps=int(self.generation_config["num_inference_steps"]),
                generator=generator,
                height=int(self.generation_config["height"]),
                width=int(self.generation_config["width"]),
                return_dict=True,
            ).images

        images, diffusion_decode_peak = self.memory_tracker.measure_span(run_pipe)
        total_peak_vram_gb = round(
            max(
                cond_reference_peak["peak_vram_gb"],
                uncond_reference_peak["peak_vram_gb"],
                prompt_conditioning_peak["peak_vram_gb"],
                diffusion_decode_peak["peak_vram_gb"],
            ),
            6,
        )
        reference_interface_metrics = build_reference_interface_metrics(
            ref_tokens_cond_bytes=self._tensor_nbytes(image_prompt_embeds),
            ref_tokens_uncond_bytes=self._tensor_nbytes(uncond_image_prompt_embeds),
            total_peak_vram_gb=total_peak_vram_gb,
            cond_reference_peak=cond_reference_peak,
            uncond_reference_peak=uncond_reference_peak,
        )
        image = _coerce_first_image(images)

        del conditioning
        del image_prompt_embeds
        del uncond_image_prompt_embeds
        self._cleanup_after_run()
        return {
            "measurement_mode": self.measurement_mode,
            "backend_load_resident_gb": self.backend_load_resident_gb,
            "component_peaks_gb": {
                "cond_reference_encode": cond_reference_peak,
                "uncond_reference_encode": uncond_reference_peak,
                "prompt_conditioning": prompt_conditioning_peak,
                "diffusion_decode": diffusion_decode_peak,
            },
            "total_peak_vram_gb": total_peak_vram_gb,
            **reference_interface_metrics,
            "ip_attention_verification": {
                "enabled": self._ip_attention_verification_enabled,
                "matched_processor_count": self._ip_attention_verification_processor_count,
                "record_count": len(self.ip_attention_verification_records),
                "records": list(self.ip_attention_verification_records),
            },
            "image": image,
        }

    def close(self) -> None:
        del self.easyref
        del self.pipe
        self._cleanup_after_run()


def _load_backend(config: Mapping[str, Any], backend: Any | None = None) -> tuple[Any, bool]:
    if backend is not None:
        return backend, False
    kind = str(config["backend"]["kind"])
    if kind == "mock":
        return MockEasyRefBackend(config), True
    return EasyRefDiffusersBackend(config), True


def run_easyref_baseline(config: Mapping[str, Any], *, backend: Any | None = None) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    ref_count = int(config["generation"]["ref_count"])
    output_dir = build_run_output_dir(config, ref_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    overwrite = bool(config["generation"]["overwrite"])
    plan = build_easyref_run_plan(config, ref_count=ref_count)

    active_backend, owns_backend = _load_backend(config, backend=backend)
    records: list[dict[str, Any]] = []
    generated_count = 0

    try:
        for spec in plan:
            if spec.image_path.is_file() and not overwrite:
                records.append(
                    {
                        "sample_id": spec.sample_id,
                        "prompt_id": spec.prompt_id,
                        "seed": spec.seed,
                        "ref_count": spec.ref_count,
                        "image_path": spec.image_local_path,
                        "generation_seconds": "0.000",
                        "status": "skipped_existing",
                    }
                )
                continue

            started_at = perf_counter()
            image = active_backend.generate_image(spec)
            duration_seconds = perf_counter() - started_at
            image.save(spec.image_path)
            generated_count += 1
            records.append(
                {
                    "sample_id": spec.sample_id,
                    "prompt_id": spec.prompt_id,
                    "seed": spec.seed,
                    "ref_count": spec.ref_count,
                    "image_path": spec.image_local_path,
                    "generation_seconds": f"{duration_seconds:.3f}",
                    "status": "generated",
                }
            )
    finally:
        close = getattr(active_backend, "close", None)
        if owns_backend and callable(close):
            close()

    return {
        "summary": {
            "suite": config["paths"]["suite"],
            "backend_name": str(getattr(active_backend, "backend_name", "unknown")),
            "output_dir": display_path(output_dir, repo_root),
            "ref_count": ref_count,
            "planned_sample_count": len(plan),
            "generated_sample_count": generated_count,
        },
        "records": records,
    }


def profile_easyref_refmem(config: Mapping[str, Any], *, backend: Any | None = None) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    output_dir = Path(config["paths"]["profile_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_counts = [int(value) for value in config["profiling"]["ref_counts"]]
    active_backend, owns_backend = _load_backend(config, backend=backend)

    records: list[dict[str, Any]] = []
    measurement_modes: set[str] = set()
    component_names: set[str] = set()

    try:
        for ref_count in ref_counts:
            for spec in build_easyref_run_plan(config, ref_count=ref_count):
                started_at = perf_counter()
                result = active_backend.profile_generation(spec)
                profiling_seconds = perf_counter() - started_at
                measurement_mode = str(result["measurement_mode"])
                measurement_modes.add(measurement_mode)
                component_names.update(result["component_peaks_gb"].keys())

                records.append(
                    {
                        "prompt_id": spec.prompt_id,
                        "seed": spec.seed,
                        "ref_count": spec.ref_count,
                        "ref_tokens_cond_bytes": int(result["ref_tokens_cond_bytes"]),
                        "ref_tokens_uncond_bytes": int(result["ref_tokens_uncond_bytes"]),
                        "persistent_ref_interface_bytes": int(result["persistent_ref_interface_bytes"]),
                        "persistent_ref_interface_gb": round(
                            float(result["persistent_ref_interface_gb"]), 6
                        ),
                        "persistent_ref_interface_share": round(
                            float(result["persistent_ref_interface_share"]), 6
                        ),
                        "reference_encode_peak_delta_gb": round(
                            float(result["reference_encode_peak_delta_gb"]), 6
                        ),
                        "reference_encode_peak_vram_gb": round(
                            float(result["reference_encode_peak_vram_gb"]), 6
                        ),
                        "total_peak_vram_gb": round(float(result["total_peak_vram_gb"]), 6),
                        "measurement_mode": measurement_mode,
                        "backend_load_resident_gb": round(float(result["backend_load_resident_gb"]), 6),
                        "component_peaks_gb": result["component_peaks_gb"],
                        "profiling_seconds": round(profiling_seconds, 6),
                        "ip_attention_verification": result.get("ip_attention_verification"),
                    }
                )
    finally:
        close = getattr(active_backend, "close", None)
        if owns_backend and callable(close):
            close()

    return {
        "summary": {
            "suite": config["paths"]["suite"],
            "backend_name": str(getattr(active_backend, "backend_name", "unknown")),
            "profile_output_dir": display_path(output_dir, repo_root),
            "run_count": len(records),
            "ref_counts": ref_counts,
            "measurement_modes": sorted(measurement_modes),
            "component_names": sorted(component_names),
        },
        "records": records,
    }


def save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
