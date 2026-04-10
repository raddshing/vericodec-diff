from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Sequence


DEFAULT_LAYER_GROUP_COUNT = 6
DEFAULT_TIMESTEP_BAND_COUNT = 4
DEFAULT_NUM_INFERENCE_STEPS = 20
DEFAULT_CANDIDATE_RANKS = (0, 2, 4, 8, 16)
DEFAULT_TARGET_MODULES = ("to_k", "to_q", "to_v", "to_out.0")
DEFAULT_LAYER_GROUP_LAYER_IDS = (
    ("down_blocks.1.attentions.0", "down_blocks.1.attentions.1"),
    ("down_blocks.2.attentions.0", "down_blocks.2.attentions.1"),
    ("mid_block.attentions.0",),
    ("up_blocks.0.attentions.0", "up_blocks.0.attentions.1"),
    ("up_blocks.0.attentions.2", "up_blocks.1.attentions.0"),
    ("up_blocks.1.attentions.1", "up_blocks.1.attentions.2"),
)
_ATTENTION_BLOCK_PATTERN = re.compile(r"^(.*?\.attentions\.\d+)(?:\.|$)")


@dataclass(frozen=True)
class LayerSpec:
    layer_id: str
    kind: str
    stage: str
    order: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "kind": self.kind,
            "stage": self.stage,
            "order": self.order,
        }


@dataclass(frozen=True)
class LayerGroup:
    group_id: str
    group_index: int
    layers: tuple[LayerSpec, ...]

    @property
    def layer_ids(self) -> tuple[str, ...]:
        return tuple(layer.layer_id for layer in self.layers)

    @property
    def is_attention_group(self) -> bool:
        return any(layer.kind == "attention" for layer in self.layers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "group_index": self.group_index,
            "layer_ids": list(self.layer_ids),
            "kinds": sorted({layer.kind for layer in self.layers}),
            "stages": sorted({layer.stage for layer in self.layers}),
            "is_attention_group": self.is_attention_group,
        }


@dataclass(frozen=True)
class TimestepBand:
    band_id: str
    band_index: int
    step_indices: tuple[int, ...]
    timestep_values: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_id": self.band_id,
            "band_index": self.band_index,
            "step_indices": list(self.step_indices),
            "timestep_values": list(self.timestep_values),
        }


@dataclass(frozen=True)
class ProbeCell:
    cell_id: str
    cell_index: int
    layer_group: LayerGroup
    timestep_band: TimestepBand
    candidate_ranks: tuple[int, ...]

    @property
    def is_attention_cell(self) -> bool:
        return self.layer_group.is_attention_group

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "cell_index": self.cell_index,
            "layer_group_id": self.layer_group.group_id,
            "timestep_band_id": self.timestep_band.band_id,
            "layer_ids": list(self.layer_group.layer_ids),
            "step_indices": list(self.timestep_band.step_indices),
            "timestep_values": list(self.timestep_band.timestep_values),
            "candidate_ranks": list(self.candidate_ranks),
            "is_attention_cell": self.is_attention_cell,
        }


@dataclass(frozen=True)
class CellSchema:
    layer_catalog: tuple[LayerSpec, ...]
    layer_groups: tuple[LayerGroup, ...]
    timestep_bands: tuple[TimestepBand, ...]
    candidate_ranks: tuple[int, ...]
    cells: tuple[ProbeCell, ...]
    timestep_values: tuple[int, ...]
    _layer_to_group: dict[str, LayerGroup] = field(repr=False, compare=False)
    _step_to_band: dict[int, TimestepBand] = field(repr=False, compare=False)
    _cell_lookup: dict[tuple[str, int], ProbeCell] = field(repr=False, compare=False)

    def locate_cell(self, *, layer_id: str, step_index: int) -> ProbeCell:
        if layer_id not in self._layer_to_group:
            raise KeyError(f"Unknown layer_id {layer_id!r}")
        if step_index not in self._step_to_band:
            raise KeyError(f"Unknown step_index {step_index!r}")
        return self._cell_lookup[(layer_id, step_index)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "layer_group_count": len(self.layer_groups),
            "timestep_band_count": len(self.timestep_bands),
            "candidate_ranks": list(self.candidate_ranks),
            "num_inference_steps": len(self.timestep_values),
            "layer_catalog": [layer.to_dict() for layer in self.layer_catalog],
            "layer_groups": [group.to_dict() for group in self.layer_groups],
            "timestep_bands": [band.to_dict() for band in self.timestep_bands],
            "cells": [cell.to_dict() for cell in self.cells],
        }


def default_layer_catalog() -> tuple[LayerSpec, ...]:
    entries = (
        ("down_blocks.0.attentions.0", "attention", "down"),
        ("down_blocks.0.attentions.1", "attention", "down"),
        ("down_blocks.1.attentions.0", "attention", "down"),
        ("down_blocks.1.attentions.1", "attention", "down"),
        ("down_blocks.2.attentions.0", "attention", "down"),
        ("down_blocks.2.attentions.1", "attention", "down"),
        ("mid_block.attentions.0", "attention", "mid"),
        ("mid_block.attentions.1", "attention", "mid"),
        ("up_blocks.0.attentions.0", "attention", "up"),
        ("up_blocks.0.attentions.1", "attention", "up"),
        ("up_blocks.0.attentions.2", "attention", "up"),
        ("up_blocks.1.attentions.0", "attention", "up"),
        ("up_blocks.1.attentions.1", "attention", "up"),
        ("up_blocks.1.attentions.2", "attention", "up"),
        ("up_blocks.2.attentions.0", "attention", "up"),
        ("up_blocks.2.attentions.1", "attention", "up"),
    )
    return tuple(
        LayerSpec(layer_id=layer_id, kind=kind, stage=stage, order=index)
        for index, (layer_id, kind, stage) in enumerate(entries)
    )


def resolve_layer_catalog(raw_catalog: Sequence[dict[str, Any]] | None) -> tuple[LayerSpec, ...]:
    if not raw_catalog:
        return default_layer_catalog()

    catalog: list[LayerSpec] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(raw_catalog):
        layer_id = str(entry["layer_id"]).strip()
        if not layer_id:
            raise ValueError("schema.layer_catalog entries must define a non-empty layer_id")
        if layer_id in seen_ids:
            raise ValueError(f"schema.layer_catalog contains duplicate layer_id {layer_id!r}")
        seen_ids.add(layer_id)
        catalog.append(
            LayerSpec(
                layer_id=layer_id,
                kind=str(entry.get("kind", "attention")).strip() or "attention",
                stage=str(entry.get("stage", "unknown")).strip() or "unknown",
                order=index,
            )
        )
    return tuple(catalog)


def resolve_candidate_ranks(raw_ranks: Sequence[int] | None) -> tuple[int, ...]:
    if raw_ranks is None:
        return DEFAULT_CANDIDATE_RANKS

    normalized = tuple(sorted({int(rank) for rank in raw_ranks}))
    if not normalized:
        raise ValueError("schema.candidate_ranks must not be empty")
    if normalized[0] != 0:
        raise ValueError("schema.candidate_ranks must include rank 0")
    if any(rank < 0 for rank in normalized):
        raise ValueError("schema.candidate_ranks must be non-negative integers")
    return normalized


def parse_candidate_ranks_argument(
    raw_value: str | Sequence[int] | None,
    *,
    default: Sequence[int] | None = None,
) -> tuple[int, ...]:
    if raw_value in (None, ""):
        return resolve_candidate_ranks(default)
    if isinstance(raw_value, str):
        items = [item.strip() for item in raw_value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in raw_value if str(item).strip()]
    return resolve_candidate_ranks([int(item) for item in items])


def resolve_timestep_values(
    *,
    timesteps: Sequence[int] | None = None,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
) -> tuple[int, ...]:
    if timesteps:
        values = tuple(int(value) for value in timesteps)
    else:
        if num_inference_steps <= 0:
            raise ValueError("schema.num_inference_steps must be positive")
        values = tuple(range(num_inference_steps - 1, -1, -1))
    if not values:
        raise ValueError("At least one timestep is required")
    return values


def _partition_sequence(length: int, partition_count: int) -> list[tuple[int, int]]:
    if partition_count <= 0:
        raise ValueError("partition_count must be positive")
    if length < partition_count:
        raise ValueError("partition_count must not exceed the number of items to partition")

    base = length // partition_count
    remainder = length % partition_count
    partitions: list[tuple[int, int]] = []
    start = 0
    for index in range(partition_count):
        width = base + (1 if index < remainder else 0)
        end = start + width
        partitions.append((start, end))
        start = end
    return partitions


def _normalize_target_modules(target_modules: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(target_module).strip() for target_module in target_modules if str(target_module).strip())
    if not normalized:
        raise ValueError("target_modules must contain at least one module suffix")
    return normalized


def _enumerate_target_module_names(unet: Any, target_modules: Sequence[str]) -> tuple[str, ...]:
    suffixes = _normalize_target_modules(target_modules)
    matched = {
        module_name
        for module_name, _module in unet.named_modules()
        if any(module_name.endswith(f".{suffix}") or module_name == suffix for suffix in suffixes)
    }
    return tuple(sorted(matched))


def discover_valid_attention_blocks(unet: Any) -> tuple[str, ...]:
    suffixes = _normalize_target_modules(DEFAULT_TARGET_MODULES)
    discovered: list[str] = []
    seen_blocks: set[str] = set()

    for module_name, _module in unet.named_modules():
        if not any(module_name.endswith(f".{suffix}") or module_name == suffix for suffix in suffixes):
            continue
        match = _ATTENTION_BLOCK_PATTERN.match(module_name)
        if match is None:
            continue
        block_name = match.group(1)
        if block_name in seen_blocks:
            continue
        seen_blocks.add(block_name)
        discovered.append(block_name)
    return tuple(discovered)


def _match_block_module_names(module_names: Sequence[str], layer_ids: Sequence[str]) -> tuple[str, ...]:
    prefixes = tuple(str(layer_id) for layer_id in layer_ids)
    return tuple(
        module_name
        for module_name in module_names
        if any(module_name.startswith(f"{prefix}.") or module_name == prefix for prefix in prefixes)
    )


def build_layer_groups(
    layer_catalog: Sequence[LayerSpec],
    *,
    group_count: int = DEFAULT_LAYER_GROUP_COUNT,
) -> tuple[LayerGroup, ...]:
    if group_count != len(DEFAULT_LAYER_GROUP_LAYER_IDS):
        raise ValueError(
            f"layer_group_count must remain {len(DEFAULT_LAYER_GROUP_LAYER_IDS)} for the SDXL UNet schema"
        )

    ordered_layers = tuple(sorted(layer_catalog, key=lambda layer: layer.order))
    layer_lookup = {layer.layer_id: layer for layer in ordered_layers}
    groups: list[LayerGroup] = []
    for group_index, layer_ids in enumerate(DEFAULT_LAYER_GROUP_LAYER_IDS):
        missing_layer_ids = [layer_id for layer_id in layer_ids if layer_id not in layer_lookup]
        if missing_layer_ids:
            raise ValueError(
                "schema.layer_catalog is missing required layer_ids for "
                f"layer_group_{group_index:02d}: {missing_layer_ids!r}"
            )
        groups.append(
            LayerGroup(
                group_id=f"layer_group_{group_index:02d}",
                group_index=group_index,
                layers=tuple(layer_lookup[layer_id] for layer_id in layer_ids),
            )
        )
    return tuple(groups)


def validate_layer_groups(
    layer_groups: Sequence[LayerGroup],
    discovered_blocks: Sequence[str],
) -> None:
    discovered = tuple(str(block).strip() for block in discovered_blocks if str(block).strip())
    if not discovered:
        raise ValueError("discovered_blocks must not be empty")

    discovered_set = set(discovered)
    assigned_blocks: set[str] = set()
    for group in layer_groups:
        if not group.layer_ids:
            raise ValueError(f"{group.group_id} must not be empty")
        for layer_id in group.layer_ids:
            if layer_id not in discovered_set:
                raise ValueError(f"{group.group_id} references block {layer_id!r}, but it does not exist in the UNet")
            if layer_id in assigned_blocks:
                raise ValueError(f"Duplicate layer_id across groups: {layer_id!r}")
            assigned_blocks.add(layer_id)


def validate_layer_group_inventory(unet: Any) -> dict[str, tuple[str, ...]]:
    module_names = _enumerate_target_module_names(unet, DEFAULT_TARGET_MODULES)
    if not module_names:
        raise ValueError(f"UNet inventory did not expose any modules for target_modules={DEFAULT_TARGET_MODULES!r}")

    groups = build_layer_groups(default_layer_catalog())
    validate_layer_groups(groups, discover_valid_attention_blocks(unet))

    inventory: dict[str, tuple[str, ...]] = {}
    for group in groups:
        for layer_id in group.layer_ids:
            matched = _match_block_module_names(module_names, (layer_id,))
            if not matched:
                raise ValueError(f"{group.group_id} references block {layer_id!r}, but it does not exist in the UNet")
            inventory[layer_id] = matched
    return inventory


def validate_cell_targets(
    schema: CellSchema,
    unet: Any,
    target_modules: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    module_names = _enumerate_target_module_names(unet, target_modules)
    if not module_names:
        raise ValueError(f"UNet inventory did not expose any modules for target_modules={tuple(target_modules)!r}")

    matches_by_cell: dict[str, tuple[str, ...]] = {}
    for cell in schema.cells:
        matched = _match_block_module_names(module_names, cell.layer_group.layer_ids)
        if not matched:
            raise ValueError(
                f"Cell {cell.cell_id} did not match any UNet modules for target_modules={tuple(target_modules)!r}"
            )
        matches_by_cell[cell.cell_id] = matched
    return matches_by_cell


def build_timestep_bands(
    timestep_values: Sequence[int],
    *,
    band_count: int = DEFAULT_TIMESTEP_BAND_COUNT,
) -> tuple[TimestepBand, ...]:
    values = tuple(int(value) for value in timestep_values)
    bands: list[TimestepBand] = []
    for band_index, (start, end) in enumerate(_partition_sequence(len(values), band_count)):
        step_indices = tuple(range(start, end))
        bands.append(
            TimestepBand(
                band_id=f"timestep_band_{band_index:02d}",
                band_index=band_index,
                step_indices=step_indices,
                timestep_values=values[start:end],
            )
        )
    return tuple(bands)


def build_cell_schema(
    *,
    layer_catalog: Sequence[LayerSpec] | None = None,
    layer_group_count: int = DEFAULT_LAYER_GROUP_COUNT,
    timestep_values: Sequence[int] | None = None,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    timestep_band_count: int = DEFAULT_TIMESTEP_BAND_COUNT,
    candidate_ranks: Sequence[int] | None = None,
    unet: Any | None = None,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
) -> CellSchema:
    catalog = tuple(layer_catalog or default_layer_catalog())
    timestep_tuple = resolve_timestep_values(
        timesteps=timestep_values,
        num_inference_steps=num_inference_steps,
    )
    ranks = resolve_candidate_ranks(candidate_ranks)
    groups = build_layer_groups(catalog, group_count=layer_group_count)
    bands = build_timestep_bands(timestep_tuple, band_count=timestep_band_count)

    layer_to_group: dict[str, LayerGroup] = {}
    for group in groups:
        for layer_id in group.layer_ids:
            layer_to_group[layer_id] = group

    step_to_band: dict[int, TimestepBand] = {}
    for band in bands:
        for step_index in band.step_indices:
            step_to_band[step_index] = band

    cells: list[ProbeCell] = []
    cell_lookup: dict[tuple[str, int], ProbeCell] = {}
    for group in groups:
        for band in bands:
            cell = ProbeCell(
                cell_id=f"{group.group_id}__{band.band_id}",
                cell_index=len(cells),
                layer_group=group,
                timestep_band=band,
                candidate_ranks=ranks,
            )
            cells.append(cell)
            for layer_id in group.layer_ids:
                for step_index in band.step_indices:
                    cell_lookup[(layer_id, step_index)] = cell

    schema = CellSchema(
        layer_catalog=catalog,
        layer_groups=groups,
        timestep_bands=bands,
        candidate_ranks=ranks,
        cells=tuple(cells),
        timestep_values=timestep_tuple,
        _layer_to_group=layer_to_group,
        _step_to_band=step_to_band,
        _cell_lookup=cell_lookup,
    )
    if unet is not None:
        validate_layer_groups(groups, discover_valid_attention_blocks(unet))
        validate_cell_targets(schema, unet, target_modules)
    return schema


__all__ = [
    "CellSchema",
    "DEFAULT_CANDIDATE_RANKS",
    "DEFAULT_LAYER_GROUP_COUNT",
    "DEFAULT_NUM_INFERENCE_STEPS",
    "DEFAULT_TARGET_MODULES",
    "DEFAULT_TIMESTEP_BAND_COUNT",
    "LayerGroup",
    "LayerSpec",
    "ProbeCell",
    "TimestepBand",
    "build_cell_schema",
    "build_layer_groups",
    "build_timestep_bands",
    "discover_valid_attention_blocks",
    "default_layer_catalog",
    "parse_candidate_ranks_argument",
    "resolve_candidate_ranks",
    "resolve_layer_catalog",
    "resolve_timestep_values",
    "validate_cell_targets",
    "validate_layer_groups",
    "validate_layer_group_inventory",
]
