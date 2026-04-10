from __future__ import annotations

from typing import Any, Mapping, Sequence

from rd_lora.cells import CellSchema, build_cell_schema


class TimestepRoutingError(ValueError):
    """Raised when manifest timestep-band routing is inconsistent with the current cell schema."""


def _schema_band_lookup(schema: CellSchema) -> dict[str, dict[str, list[int]]]:
    return {
        band.band_id: {
            "step_indices": list(band.step_indices),
            "timestep_values": list(band.timestep_values),
        }
        for band in schema.timestep_bands
    }


def build_timestep_band_routes(
    timestep_bands: Sequence[str],
    *,
    schema: CellSchema | None = None,
) -> dict[str, dict[str, list[int]]]:
    active_schema = schema or build_cell_schema()
    available = _schema_band_lookup(active_schema)
    routes: dict[str, dict[str, list[int]]] = {}
    for raw_band_name in timestep_bands:
        band_name = str(raw_band_name).strip()
        if band_name not in available:
            raise TimestepRoutingError(
                f"Unknown timestep_band {band_name!r}; available={sorted(available)}"
            )
        routes[band_name] = {
            "step_indices": list(available[band_name]["step_indices"]),
            "timestep_values": list(available[band_name]["timestep_values"]),
        }
    return routes


def build_adapter_routing_table(
    routes: Mapping[str, Mapping[str, Sequence[int]]],
    *,
    band_to_adapter: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for band_name, route in routes.items():
        adapter_name = str(band_to_adapter.get(str(band_name), "")).strip()
        if not adapter_name:
            raise TimestepRoutingError(f"Missing adapter_name for timestep_band {band_name!r}")
        table[str(band_name)] = {
            "adapter_name": adapter_name,
            "step_indices": [int(value) for value in route["step_indices"]],
            "timestep_values": [int(value) for value in route["timestep_values"]],
        }
    return table


def resolve_timestep_band_name(
    routing_table: Mapping[str, Mapping[str, Sequence[int]]],
    *,
    step_index: int | None = None,
    timestep_value: int | None = None,
) -> str:
    provided = int(step_index is not None) + int(timestep_value is not None)
    if provided != 1:
        raise TimestepRoutingError("Specify exactly one of step_index or timestep_value")

    matches: list[str] = []
    for band_name, route in routing_table.items():
        values = route["step_indices"] if step_index is not None else route["timestep_values"]
        candidate = int(step_index if step_index is not None else timestep_value)
        if candidate in {int(value) for value in values}:
            matches.append(str(band_name))

    if len(matches) != 1:
        candidate_name = "step_index" if step_index is not None else "timestep_value"
        candidate_value = step_index if step_index is not None else timestep_value
        raise TimestepRoutingError(
            f"{candidate_name}={candidate_value} mapped to {len(matches)} bands; matches={matches}"
        )
    return matches[0]


def resolve_active_adapter_name(
    routing_table: Mapping[str, Mapping[str, Sequence[int] | str]],
    *,
    step_index: int | None = None,
    timestep_value: int | None = None,
) -> str:
    band_name = resolve_timestep_band_name(
        routing_table,
        step_index=step_index,
        timestep_value=timestep_value,
    )
    adapter_name = str(routing_table[band_name].get("adapter_name", "")).strip()
    if not adapter_name:
        raise TimestepRoutingError(f"Routing entry for {band_name!r} is missing adapter_name")
    return adapter_name


def project_training_timestep_to_step_index(
    timestep_value: int,
    *,
    num_train_timesteps: int,
    reference_step_count: int,
) -> int:
    if num_train_timesteps <= 0:
        raise TimestepRoutingError("num_train_timesteps must be positive")
    if reference_step_count <= 0:
        raise TimestepRoutingError("reference_step_count must be positive")

    timestep = int(timestep_value)
    max_timestep = max(1, int(num_train_timesteps) - 1)
    if timestep < 0 or timestep > max_timestep:
        raise TimestepRoutingError(
            f"timestep_value={timestep} must be in [0, {max_timestep}]"
        )

    progress = float(max_timestep - timestep) / float(max_timestep)
    projected = int(round(progress * float(reference_step_count - 1)))
    return max(0, min(reference_step_count - 1, projected))


def resolve_timestep_band_name_for_training_timesteps(
    routing_table: Mapping[str, Mapping[str, Sequence[int] | str]],
    *,
    timesteps: Sequence[int],
    num_train_timesteps: int,
    reference_step_count: int = 20,
) -> str:
    if not timesteps:
        raise TimestepRoutingError("timesteps must not be empty")

    matches = {
        resolve_timestep_band_name(
            routing_table,
            step_index=project_training_timestep_to_step_index(
                int(timestep_value),
                num_train_timesteps=num_train_timesteps,
                reference_step_count=reference_step_count,
            ),
        )
        for timestep_value in timesteps
    }
    if len(matches) != 1:
        raise TimestepRoutingError(
            f"timesteps mapped to multiple bands: {sorted(matches)}"
        )
    return next(iter(matches))


def resolve_active_adapter_name_for_training_timesteps(
    routing_table: Mapping[str, Mapping[str, Sequence[int] | str]],
    *,
    timesteps: Sequence[int],
    num_train_timesteps: int,
    reference_step_count: int = 20,
) -> str:
    band_name = resolve_timestep_band_name_for_training_timesteps(
        routing_table,
        timesteps=timesteps,
        num_train_timesteps=num_train_timesteps,
        reference_step_count=reference_step_count,
    )
    adapter_name = str(routing_table[band_name].get("adapter_name", "")).strip()
    if not adapter_name:
        raise TimestepRoutingError(f"Routing entry for {band_name!r} is missing adapter_name")
    return adapter_name


def describe_routing_table(routing_table: Mapping[str, Mapping[str, Sequence[int] | str]]) -> dict[str, Any]:
    return {
        "timestep_bands": list(routing_table),
        "mapped_timestep_total": sum(len(route["step_indices"]) for route in routing_table.values()),
        "adapter_names": {
            str(band_name): str(route["adapter_name"])
            for band_name, route in routing_table.items()
        },
    }


__all__ = [
    "TimestepRoutingError",
    "build_adapter_routing_table",
    "build_timestep_band_routes",
    "describe_routing_table",
    "project_training_timestep_to_step_index",
    "resolve_active_adapter_name_for_training_timesteps",
    "resolve_active_adapter_name",
    "resolve_timestep_band_name",
    "resolve_timestep_band_name_for_training_timesteps",
]
