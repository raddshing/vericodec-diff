from __future__ import annotations

from typing import Any, Mapping, Sequence

from rd_lora.cells import build_cell_schema


class TimestepRoutingError(ValueError):
    """Raised when timestep-band routing metadata is inconsistent."""


def build_timestep_band_routes(timestep_bands: Sequence[str] | None = None) -> dict[str, list[int]]:
    schema = build_cell_schema()
    available = {band.band_id: list(band.step_indices) for band in schema.timestep_bands}
    if timestep_bands is None:
        return available
    routes: dict[str, list[int]] = {}
    for band_id in timestep_bands:
        normalized = str(band_id).strip()
        if normalized not in available:
            raise TimestepRoutingError(f"Unknown timestep band {normalized!r}")
        routes[normalized] = list(available[normalized])
    return routes


def build_step_to_band_map(routes: Mapping[str, Sequence[int]]) -> dict[int, str]:
    step_to_band: dict[int, str] = {}
    for band_id, step_indices in routes.items():
        for step_index in step_indices:
            parsed = int(step_index)
            if parsed in step_to_band:
                raise TimestepRoutingError(f"Step index {parsed} is assigned to multiple timestep bands")
            step_to_band[parsed] = str(band_id)
    return step_to_band


def build_adapter_step_map(routes: Mapping[str, Sequence[int]], *, band_to_adapter: Mapping[str, str]) -> dict[int, str]:
    step_to_band = build_step_to_band_map(routes)
    adapter_step_map: dict[int, str] = {}
    for step_index, band_id in step_to_band.items():
        if band_id not in band_to_adapter:
            raise TimestepRoutingError(f"Missing adapter for timestep band {band_id!r}")
        adapter_step_map[step_index] = str(band_to_adapter[band_id])
    return adapter_step_map


def summarize_routes(routes: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    step_to_band = build_step_to_band_map(routes)
    return {
        "timestep_bands": sorted(str(key) for key in routes),
        "step_count": len(step_to_band),
        "step_to_band": {str(step): band for step, band in sorted(step_to_band.items())},
    }


__all__ = [
    "TimestepRoutingError",
    "build_adapter_step_map",
    "build_step_to_band_map",
    "build_timestep_band_routes",
    "summarize_routes",
]
