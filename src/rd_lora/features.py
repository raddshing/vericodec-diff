from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REQUIRED_PROBE_ROW_FIELDNAMES = (
    "task",
    "cell_id",
    "layer_group",
    "timestep_band",
    "candidate_rank",
    "pre_loss",
    "post_loss",
    "utility",
    "optimizer_steps",
    "train_batch_count",
    "val_batch_count",
)

# Keep compatibility aliases alongside the required stage-D probe fields.
UTILITY_RECORD_FIELDNAMES = REQUIRED_PROBE_ROW_FIELDNAMES + (
    "layer_group_id",
    "timestep_band_id",
    "utility_score",
)

REQUIRED_PROBE_SUMMARY_KEYS = (
    "probe_mode",
    "task",
    "cell_count",
    "candidate_ranks",
    "row_count",
    "loaded_components",
    "used_gpu",
    "peak_vram_mib",
)

REQUIRED_PROVENANCE_KEYS = (
    "run_mode",
    "used_gpu",
    "peak_vram_mib",
    "python",
    "torch",
    "torch_cuda_is_available",
    "torch_device_count",
    "gpu_names",
    "diffusers",
    "diffusers_file",
)


def _round_float(value: float | int, *, digits: int = 6) -> float:
    return round(float(value), digits)


def build_probe_row(
    *,
    task: str,
    cell_id: str,
    layer_group: str,
    timestep_band: str,
    candidate_rank: int,
    pre_loss: float,
    post_loss: float,
    optimizer_steps: int,
    train_batch_count: int,
    val_batch_count: int,
) -> dict[str, Any]:
    utility = float(pre_loss) - float(post_loss)
    return {
        "task": str(task),
        "cell_id": str(cell_id),
        "layer_group": str(layer_group),
        "timestep_band": str(timestep_band),
        "candidate_rank": int(candidate_rank),
        "pre_loss": _round_float(pre_loss),
        "post_loss": _round_float(post_loss),
        "utility": _round_float(utility),
        "optimizer_steps": int(optimizer_steps),
        "train_batch_count": int(train_batch_count),
        "val_batch_count": int(val_batch_count),
        "layer_group_id": str(layer_group),
        "timestep_band_id": str(timestep_band),
        "utility_score": _round_float(utility),
    }


def build_probe_summary(
    *,
    probe_mode: str,
    task: str,
    cell_count: int,
    candidate_ranks: Sequence[int],
    row_count: int,
    loaded_components: Sequence[str],
    used_gpu: bool,
    peak_vram_mib: float | int | None,
) -> dict[str, Any]:
    return {
        "probe_mode": str(probe_mode),
        "task": str(task),
        "cell_count": int(cell_count),
        "candidate_ranks": [int(rank) for rank in candidate_ranks],
        "row_count": int(row_count),
        "loaded_components": [str(component) for component in loaded_components],
        "used_gpu": bool(used_gpu),
        "peak_vram_mib": 0.0 if peak_vram_mib in (None, "") else _round_float(float(peak_vram_mib), digits=3),
    }


def collect_run_provenance(
    *,
    run_mode: str,
    used_gpu: bool,
    peak_vram_mib: float | int | None,
) -> dict[str, Any]:
    import diffusers  # type: ignore
    import torch  # type: ignore

    gpu_names = []
    device_count = 0
    cuda_is_available = bool(torch.cuda.is_available())
    if cuda_is_available:
        device_count = int(torch.cuda.device_count())
        for index in range(device_count):
            gpu_names.append(str(torch.cuda.get_device_name(index)))

    return {
        "run_mode": str(run_mode),
        "used_gpu": bool(used_gpu),
        "peak_vram_mib": 0.0 if peak_vram_mib in (None, "") else _round_float(float(peak_vram_mib), digits=3),
        "python": sys.version.split()[0],
        "torch": str(getattr(torch, "__version__", "unknown")),
        "torch_cuda_is_available": cuda_is_available,
        "torch_device_count": device_count,
        "gpu_names": gpu_names,
        "diffusers": str(getattr(diffusers, "__version__", "unknown")),
        "diffusers_file": str(Path(getattr(diffusers, "__file__", "")).resolve()),
        "schema_version": "1.0",
        "torch_cuda_version": str(getattr(torch.version, "cuda", "N/A")),
        "accelerate_config_file": "",
        "git_commit": "",
        "backend": "probe",
        "task": "",
        "allocation_manifest": "",
        "timestamp_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }


def validate_probe_payloads(
    *,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    for row in rows:
        missing = [field for field in REQUIRED_PROBE_ROW_FIELDNAMES if field not in row]
        if missing:
            raise ValueError(f"Probe row is missing required fields: {missing}")
    missing_summary = [field for field in REQUIRED_PROBE_SUMMARY_KEYS if field not in summary]
    if missing_summary:
        raise ValueError(f"Probe summary is missing required fields: {missing_summary}")
    missing_provenance = [field for field in REQUIRED_PROVENANCE_KEYS if field not in provenance]
    if missing_provenance:
        raise ValueError(f"Run provenance is missing required fields: {missing_provenance}")


def save_json_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "REQUIRED_PROBE_ROW_FIELDNAMES",
    "REQUIRED_PROBE_SUMMARY_KEYS",
    "REQUIRED_PROVENANCE_KEYS",
    "UTILITY_RECORD_FIELDNAMES",
    "build_probe_row",
    "build_probe_summary",
    "collect_run_provenance",
    "save_json_payload",
    "validate_probe_payloads",
]
