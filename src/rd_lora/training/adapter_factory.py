from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from rd_lora.cells import build_cell_schema
from rd_lora.runtime.allocation_manifest import load_allocation_manifest, validate_allocation_manifest
from rd_lora.training.timestep_routing import build_adapter_step_map, build_timestep_band_routes, summarize_routes


DEFAULT_TARGET_MODULES = ("to_k", "to_q", "to_v", "to_out.0")


class AdapterFactoryError(ValueError):
    """Raised when a manifest cannot be converted into backend adapter specs."""


def _layer_group_module_map(target_modules: Sequence[str]) -> dict[str, list[str]]:
    schema = build_cell_schema()
    mapping: dict[str, list[str]] = {}
    for group in schema.layer_groups:
        module_names: list[str] = []
        for layer_id in group.layer_ids:
            for target_module in target_modules:
                module_names.append(f"{layer_id}.{target_module}")
        mapping[group.group_id] = module_names
    return mapping


def _cells_by_field(cells: Sequence[Mapping[str, Any]], field_name: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault(str(cell[field_name]), []).append(dict(cell))
    return grouped


def _rank_alpha_maps(
    *,
    cells: Sequence[Mapping[str, Any]],
    target_modules: Sequence[str],
) -> tuple[dict[str, int], dict[str, int]]:
    module_lookup = _layer_group_module_map(target_modules)
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}
    for cell in cells:
        layer_group = str(cell["layer_group"])
        for module_name in module_lookup[layer_group]:
            rank_pattern[module_name] = int(cell["rank"])
            alpha_pattern[module_name] = int(cell["alpha"])
    return rank_pattern, alpha_pattern


def build_backend_adapter_plan(
    backend: str,
    manifest: Mapping[str, Any] | str | Path,
    *,
    use_rslora: bool = True,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
) -> dict[str, Any]:
    if isinstance(manifest, (str, Path)):
        normalized_manifest = load_allocation_manifest(manifest)
        manifest_path = str(Path(manifest).expanduser().resolve())
    else:
        normalized_manifest = validate_allocation_manifest(manifest)
        manifest_path = None

    normalized_backend = str(backend).strip()
    if normalized_backend != normalized_manifest["backend"]:
        raise AdapterFactoryError(
            f"Requested backend {normalized_backend!r} does not match manifest backend {normalized_manifest['backend']!r}"
        )

    target_module_list = [str(value) for value in target_modules]
    if tuple(target_module_list) != DEFAULT_TARGET_MODULES:
        raise AdapterFactoryError(
            f"LoRA target_modules must remain {DEFAULT_TARGET_MODULES!r} for RD-LoRA Stage D"
        )

    cells = list(normalized_manifest["cells"])
    adapter_banks: list[dict[str, Any]] = []

    if normalized_backend == "uniform":
        ranks = {int(cell["rank"]) for cell in cells}
        alphas = {int(cell["alpha"]) for cell in cells}
        if len(ranks) != 1 or len(alphas) != 1:
            raise AdapterFactoryError("uniform backend requires a single shared rank and alpha")
        rank_pattern, alpha_pattern = _rank_alpha_maps(cells=cells, target_modules=target_module_list)
        adapter_banks.append(
            {
                "adapter_name": "uniform_bank",
                "timestep_band": None,
                "rank_pattern": rank_pattern,
                "alpha_pattern": alpha_pattern,
            }
        )

    elif normalized_backend == "layer_only":
        by_group = _cells_by_field(cells, "layer_group")
        collapsed_cells: list[dict[str, Any]] = []
        for layer_group, group_cells in sorted(by_group.items()):
            ranks = {int(cell["rank"]) for cell in group_cells}
            alphas = {int(cell["alpha"]) for cell in group_cells}
            if len(ranks) != 1 or len(alphas) != 1:
                raise AdapterFactoryError(
                    f"layer_only backend requires constant rank/alpha across timestep bands for {layer_group!r}"
                )
            collapsed_cells.append(
                {
                    "cell_id": layer_group,
                    "layer_group": layer_group,
                    "timestep_band": "all",
                    "rank": int(group_cells[0]["rank"]),
                    "alpha": int(group_cells[0]["alpha"]),
                    "target_modules": list(target_module_list),
                    "adapter_name": "layer_only_bank",
                }
            )
        rank_pattern, alpha_pattern = _rank_alpha_maps(cells=collapsed_cells, target_modules=target_module_list)
        adapter_banks.append(
            {
                "adapter_name": "layer_only_bank",
                "timestep_band": None,
                "rank_pattern": rank_pattern,
                "alpha_pattern": alpha_pattern,
            }
        )

    elif normalized_backend in {"timestep_only", "proposed"}:
        by_band = _cells_by_field(cells, "timestep_band")
        for timestep_band in normalized_manifest["timestep_bands"]:
            band_cells = by_band[str(timestep_band)]
            if normalized_backend == "timestep_only":
                ranks = {int(cell["rank"]) for cell in band_cells}
                alphas = {int(cell["alpha"]) for cell in band_cells}
                if len(ranks) != 1 or len(alphas) != 1:
                    raise AdapterFactoryError(
                        f"timestep_only backend requires constant rank/alpha across layers for {timestep_band!r}"
                    )
            rank_pattern, alpha_pattern = _rank_alpha_maps(cells=band_cells, target_modules=target_module_list)
            adapter_banks.append(
                {
                    "adapter_name": f"{normalized_backend}__{timestep_band}",
                    "timestep_band": str(timestep_band),
                    "rank_pattern": rank_pattern,
                    "alpha_pattern": alpha_pattern,
                }
            )

    else:
        raise AdapterFactoryError(f"Unsupported backend {normalized_backend!r}")

    routes = build_timestep_band_routes(normalized_manifest["timestep_bands"])
    band_to_adapter = {
        str(bank["timestep_band"]): str(bank["adapter_name"])
        for bank in adapter_banks
        if bank["timestep_band"] is not None
    }
    adapter_step_map = build_adapter_step_map(routes, band_to_adapter=band_to_adapter) if band_to_adapter else {}
    bootstrap_rank = max(int(cell["rank"]) for cell in cells) if cells else 0

    return {
        "schema_version": "1.0",
        "manifest_path": manifest_path,
        "backend": normalized_backend,
        "use_rslora": bool(use_rslora),
        "target_modules": list(target_module_list),
        "bootstrap_rank": max(1, bootstrap_rank) if bootstrap_rank > 0 else 1,
        "adapter_banks": adapter_banks,
        "routing": {
            "band_routes": routes,
            "summary": summarize_routes(routes),
            "adapter_step_map": {str(step): adapter for step, adapter in sorted(adapter_step_map.items())},
        },
        "rank_budget_total": int(normalized_manifest["rank_budget_total"]),
    }


def write_adapter_plan(plan: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


__all__ = [
    "AdapterFactoryError",
    "DEFAULT_TARGET_MODULES",
    "build_backend_adapter_plan",
    "write_adapter_plan",
]
