from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ALLOCATION_MANIFEST_SCHEMA_VERSION = "1.0"
_REQUIRED_TOP_LEVEL_KEYS = (
    "schema_version",
    "backend",
    "rank_budget_total",
    "layer_groups",
    "timestep_bands",
    "cells",
)
_REQUIRED_CELL_KEYS = (
    "cell_id",
    "layer_group",
    "timestep_band",
    "rank",
    "alpha",
    "target_modules",
    "adapter_name",
)


class AllocationManifestError(ValueError):
    """Raised when an RD-LoRA allocation manifest is malformed."""


def _require_mapping(obj: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(obj, Mapping):
        raise AllocationManifestError(f"{name} must be a mapping")
    return dict(obj)


def _normalize_string_list(values: Any, *, name: str) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise AllocationManifestError(f"{name} must be a list")
    normalized = [str(value).strip() for value in values]
    if any(not value for value in normalized):
        raise AllocationManifestError(f"{name} must not contain empty entries")
    return normalized


def validate_allocation_manifest(obj: Mapping[str, Any]) -> dict[str, Any]:
    payload = _require_mapping(obj, name="allocation manifest")
    missing = [key for key in _REQUIRED_TOP_LEVEL_KEYS if key not in payload]
    if missing:
        raise AllocationManifestError(f"allocation manifest is missing keys: {missing}")
    if str(payload["schema_version"]) != ALLOCATION_MANIFEST_SCHEMA_VERSION:
        raise AllocationManifestError(
            f"schema_version must be {ALLOCATION_MANIFEST_SCHEMA_VERSION!r}"
        )

    backend = str(payload["backend"]).strip()
    if backend not in {"uniform", "layer_only", "timestep_only", "proposed"}:
        raise AllocationManifestError(f"Unsupported backend {backend!r}")
    rank_budget_total = int(payload["rank_budget_total"])
    if rank_budget_total < 0:
        raise AllocationManifestError("rank_budget_total must be non-negative")

    layer_groups = _normalize_string_list(payload["layer_groups"], name="layer_groups")
    timestep_bands = _normalize_string_list(payload["timestep_bands"], name="timestep_bands")
    cells_value = payload["cells"]
    if not isinstance(cells_value, Sequence) or isinstance(cells_value, (str, bytes)):
        raise AllocationManifestError("cells must be a list")
    cells: list[dict[str, Any]] = []
    seen_cell_ids: set[str] = set()
    total_rank = 0
    for index, cell_value in enumerate(cells_value):
        cell = _require_mapping(cell_value, name=f"cells[{index}]")
        missing_cell_keys = [key for key in _REQUIRED_CELL_KEYS if key not in cell]
        if missing_cell_keys:
            raise AllocationManifestError(f"cells[{index}] is missing keys: {missing_cell_keys}")
        cell_id = str(cell["cell_id"]).strip()
        if not cell_id:
            raise AllocationManifestError(f"cells[{index}].cell_id must be non-empty")
        if cell_id in seen_cell_ids:
            raise AllocationManifestError(f"Duplicate cell_id {cell_id!r}")
        seen_cell_ids.add(cell_id)

        layer_group = str(cell["layer_group"]).strip()
        timestep_band = str(cell["timestep_band"]).strip()
        if layer_group not in layer_groups:
            raise AllocationManifestError(f"Unknown layer_group {layer_group!r} for cell {cell_id!r}")
        if timestep_band not in timestep_bands:
            raise AllocationManifestError(f"Unknown timestep_band {timestep_band!r} for cell {cell_id!r}")

        rank = int(cell["rank"])
        alpha = int(cell["alpha"])
        if rank < 0:
            raise AllocationManifestError(f"rank must be non-negative for cell {cell_id!r}")
        if alpha < 0:
            raise AllocationManifestError(f"alpha must be non-negative for cell {cell_id!r}")
        target_modules = _normalize_string_list(cell["target_modules"], name=f"cells[{index}].target_modules")
        adapter_name = str(cell["adapter_name"]).strip()
        if not adapter_name:
            raise AllocationManifestError(f"adapter_name must be non-empty for cell {cell_id!r}")

        total_rank += rank
        cells.append(
            {
                "cell_id": cell_id,
                "layer_group": layer_group,
                "timestep_band": timestep_band,
                "rank": rank,
                "alpha": alpha,
                "target_modules": target_modules,
                "adapter_name": adapter_name,
            }
        )

    if not cells:
        raise AllocationManifestError("cells must not be empty")
    if total_rank > rank_budget_total:
        raise AllocationManifestError(
            f"Selected rank total {total_rank} exceeds rank_budget_total {rank_budget_total}"
        )

    return {
        "schema_version": ALLOCATION_MANIFEST_SCHEMA_VERSION,
        "backend": backend,
        "rank_budget_total": rank_budget_total,
        "layer_groups": layer_groups,
        "timestep_bands": timestep_bands,
        "cells": cells,
    }


def load_allocation_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return validate_allocation_manifest(_require_mapping(payload, name=str(manifest_path)))


def write_allocation_manifest(obj: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    payload = validate_allocation_manifest(obj)
    manifest_path = Path(path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "ALLOCATION_MANIFEST_SCHEMA_VERSION",
    "AllocationManifestError",
    "load_allocation_manifest",
    "validate_allocation_manifest",
    "write_allocation_manifest",
]
