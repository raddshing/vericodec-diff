from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROVENANCE_SCHEMA_VERSION = "1.0"
FORBIDDEN_SOURCE_TOKENS = (
    "mock",
    "smoke",
    "cpu_smoke",
    "cpu_debug",
    "stageb_cpu_smoke",
    "stagec_cpu_smoke",
)
_REQUIRED_KEYS = (
    "schema_version",
    "run_mode",
    "used_gpu",
    "python",
    "torch",
    "torch_cuda_is_available",
    "torch_cuda_version",
    "torch_device_count",
    "gpu_names",
    "diffusers",
    "diffusers_file",
    "accelerate_config_file",
    "git_commit",
    "backend",
    "task",
    "allocation_manifest",
    "peak_vram_mib",
    "timestamp_utc",
)


class ProvenanceError(RuntimeError):
    """Raised when runtime provenance is missing or invalid."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ProvenanceError(f"Unable to resolve git commit for {repo_root}")
    commit = completed.stdout.strip()
    if not commit:
        raise ProvenanceError(f"Empty git commit for {repo_root}")
    return commit


def _to_string_path(value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    return str(Path(value).expanduser().resolve())


def _collect_gpu_names(torch_module: Any, device_count: int) -> list[str]:
    names: list[str] = []
    for index in range(device_count):
        try:
            names.append(str(torch_module.cuda.get_device_name(index)))
        except Exception:
            names.append(f"cuda:{index}")
    return names


def query_peak_vram_mib() -> float | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None

    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return None
    try:
        if not bool(cuda.is_available()):
            return None
    except Exception:
        return None

    max_memory_allocated = getattr(cuda, "max_memory_allocated", None)
    if callable(max_memory_allocated):
        try:
            raw_bytes = float(max_memory_allocated())
        except Exception:
            raw_bytes = 0.0
        if raw_bytes > 0.0:
            return round(raw_bytes / float(1024**2), 3)
    return None


def collect_runtime_metadata(
    *,
    repo_root: str | Path,
    accelerate_config_file: str | Path,
    require_cuda: bool,
    expected_diffusers_substring: str | None = None,
) -> dict[str, Any]:
    repo_root_path = Path(repo_root).expanduser().resolve()
    accelerate_config_path = Path(accelerate_config_file).expanduser().resolve()
    if not accelerate_config_path.is_file():
        raise ProvenanceError(f"Accelerate config not found: {accelerate_config_path}")

    try:
        import torch  # type: ignore
    except Exception as exc:
        raise ProvenanceError(f"Unable to import torch: {type(exc).__name__}: {exc}") from exc

    try:
        import diffusers  # type: ignore
    except Exception as exc:
        raise ProvenanceError(f"Unable to import diffusers: {type(exc).__name__}: {exc}") from exc

    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        raise ProvenanceError("torch.cuda is unavailable")

    torch_cuda_is_available = bool(cuda.is_available())
    torch_device_count = int(cuda.device_count())
    if require_cuda and not torch_cuda_is_available:
        raise ProvenanceError("torch.cuda.is_available() must be True for a real_gpu run")
    if torch_device_count < 1:
        raise ProvenanceError("torch.cuda.device_count() must be at least 1")

    diffusers_file = str(Path(getattr(diffusers, "__file__", "")).resolve())
    expected_substring = (expected_diffusers_substring or "").strip()
    if expected_substring and expected_substring not in diffusers_file:
        raise ProvenanceError(
            f"Imported diffusers path must contain {expected_substring!r}, found {diffusers_file!r}"
        )

    torch_version = str(getattr(torch, "__version__", "unknown"))
    torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    diffusers_version = str(getattr(diffusers, "__version__", "unknown"))
    gpu_names = _collect_gpu_names(torch, torch_device_count)

    return {
        "python": sys.version.split()[0],
        "torch": torch_version,
        "torch_cuda_is_available": torch_cuda_is_available,
        "torch_cuda_version": None if torch_cuda_version in (None, "") else str(torch_cuda_version),
        "torch_device_count": torch_device_count,
        "gpu_names": gpu_names,
        "diffusers": diffusers_version,
        "diffusers_file": diffusers_file,
        "accelerate_config_file": str(accelerate_config_path),
        "git_commit": _git_commit(repo_root_path),
        "timestamp_utc": _utc_now_iso(),
    }


def validate_provenance_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise ProvenanceError(f"run_provenance.json is missing keys: {missing}")
    if str(payload["schema_version"]) != PROVENANCE_SCHEMA_VERSION:
        raise ProvenanceError(
            f"Unsupported provenance schema_version {payload['schema_version']!r}; "
            f"expected {PROVENANCE_SCHEMA_VERSION!r}"
        )
    if str(payload["run_mode"]).strip() not in {"real_gpu", "mock"}:
        raise ProvenanceError("run_mode must be 'real_gpu' or 'mock'")
    if not isinstance(payload["gpu_names"], list):
        raise ProvenanceError("gpu_names must be a list")
    normalized = dict(payload)
    normalized["torch_device_count"] = int(payload["torch_device_count"])
    if payload["peak_vram_mib"] not in (None, ""):
        normalized["peak_vram_mib"] = float(payload["peak_vram_mib"])
    else:
        normalized["peak_vram_mib"] = None
    return normalized


def load_run_provenance(path: str | Path) -> dict[str, Any]:
    provenance_path = Path(path).expanduser().resolve()
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProvenanceError(f"{provenance_path} must parse to a JSON object")
    return validate_provenance_payload(payload)


def iter_recorded_source_paths(payload: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("source_run_dir", "source_probe_dir", "source_surrogate_dir", "source_allocation_dir"):
        value = payload.get(key)
        if value not in (None, ""):
            paths.append(str(value))
    source_run_dirs = payload.get("source_run_dirs")
    if isinstance(source_run_dirs, Sequence) and not isinstance(source_run_dirs, (str, bytes)):
        for value in source_run_dirs:
            if value not in (None, ""):
                paths.append(str(value))
    return paths


def path_contains_forbidden_token(path_value: str | Path, tokens: Iterable[str] = FORBIDDEN_SOURCE_TOKENS) -> bool:
    lower_value = str(path_value).lower()
    return any(str(token).lower() in lower_value for token in tokens)


def assert_real_gpu_provenance(
    payload: Mapping[str, Any],
    *,
    require_gpu: bool,
    forbid_mock: bool,
) -> dict[str, Any]:
    normalized = validate_provenance_payload(payload)
    if normalized["run_mode"] != "real_gpu":
        raise ProvenanceError("run_mode must be 'real_gpu'")
    if require_gpu:
        if not bool(normalized["used_gpu"]):
            raise ProvenanceError("used_gpu must be true")
        if not bool(normalized["torch_cuda_is_available"]):
            raise ProvenanceError("torch_cuda_is_available must be true")
        if int(normalized["torch_device_count"]) < 1:
            raise ProvenanceError("torch_device_count must be at least 1")
        peak_vram_mib = normalized["peak_vram_mib"]
        if peak_vram_mib is None or float(peak_vram_mib) <= 0.0:
            raise ProvenanceError("peak_vram_mib must exist and be > 0")
    if forbid_mock:
        for path_value in iter_recorded_source_paths(normalized):
            if path_contains_forbidden_token(path_value):
                raise ProvenanceError(f"Forbidden mock/smoke token found in source path: {path_value}")
    return normalized


def assert_real_gpu_training_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    return assert_real_gpu_provenance(
        payload,
        require_gpu=True,
        forbid_mock=False,
    )


def write_run_provenance(
    *,
    run_dir: str | Path,
    repo_root: str | Path,
    run_mode: str,
    used_gpu: bool,
    backend: str,
    task: str,
    accelerate_config_file: str | Path,
    allocation_manifest: str | Path | None,
    peak_vram_mib: float | int | None,
    require_cuda: bool,
    expected_diffusers_substring: str | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = collect_runtime_metadata(
        repo_root=repo_root,
        accelerate_config_file=accelerate_config_file,
        require_cuda=require_cuda,
        expected_diffusers_substring=expected_diffusers_substring,
    )
    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "run_mode": str(run_mode),
        "used_gpu": bool(used_gpu),
        "python": runtime["python"],
        "torch": runtime["torch"],
        "torch_cuda_is_available": runtime["torch_cuda_is_available"],
        "torch_cuda_version": runtime["torch_cuda_version"],
        "torch_device_count": runtime["torch_device_count"],
        "gpu_names": runtime["gpu_names"],
        "diffusers": runtime["diffusers"],
        "diffusers_file": runtime["diffusers_file"],
        "accelerate_config_file": runtime["accelerate_config_file"],
        "git_commit": runtime["git_commit"],
        "backend": str(backend),
        "task": str(task),
        "allocation_manifest": _to_string_path(allocation_manifest),
        "peak_vram_mib": None if peak_vram_mib in (None, "") else round(float(peak_vram_mib), 3),
        "timestamp_utc": runtime["timestamp_utc"],
    }
    if extra_fields:
        payload.update(dict(extra_fields))
    validate_provenance_payload(payload)
    output_path = Path(run_dir).expanduser().resolve() / "run_provenance.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "FORBIDDEN_SOURCE_TOKENS",
    "PROVENANCE_SCHEMA_VERSION",
    "ProvenanceError",
    "assert_real_gpu_provenance",
    "assert_real_gpu_training_provenance",
    "collect_runtime_metadata",
    "iter_recorded_source_paths",
    "load_run_provenance",
    "path_contains_forbidden_token",
    "query_peak_vram_mib",
    "validate_provenance_payload",
    "write_run_provenance",
]
