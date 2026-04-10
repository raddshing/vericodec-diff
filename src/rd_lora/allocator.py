from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rd_lora.cells import CellSchema, DEFAULT_TARGET_MODULES, build_cell_schema
from rd_lora.runtime.allocation_manifest import load_allocation_manifest as _load_runtime_allocation_manifest


CANONICAL_SCORED_ROW_FIELDNAMES = (
    "cell_id",
    "layer_group",
    "timestep_band",
    "candidate_rank",
    "predicted_utility",
)


class AllocationValidationError(ValueError):
    """Raised when canonical allocation inputs or outputs are invalid."""


@dataclass(frozen=True)
class _DpState:
    utility_scaled: int
    used_budget: int
    signature: tuple[int, ...]
    selections: tuple[dict[str, Any], ...]


def _as_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def _round_float(value: float, *, digits: int = 6) -> float:
    return round(float(value), int(digits))


def _as_non_negative_int(value: Any, *, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise AllocationValidationError(f"{name} must be non-negative")
    return parsed


def _candidate_ranks_from_records(records: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    candidate_ranks = sorted({int(record["candidate_rank"]) for record in records})
    if not candidate_ranks:
        raise AllocationValidationError("At least one candidate rank is required")
    if candidate_ranks[0] != 0:
        raise AllocationValidationError("candidate ranks must include rank 0")
    return tuple(candidate_ranks)


def build_allocation_schema(
    records: Sequence[Mapping[str, Any]],
    *,
    schema: CellSchema | None = None,
) -> CellSchema:
    if schema is not None:
        return schema
    return build_cell_schema(candidate_ranks=_candidate_ranks_from_records(records))


def _resolve_layer_group(row: Mapping[str, Any], schema: CellSchema) -> str:
    raw_layer_group = row.get("layer_group")
    if isinstance(raw_layer_group, str):
        layer_group = raw_layer_group.strip()
        if not layer_group:
            raise AllocationValidationError("layer_group must be non-empty when provided as a string")
        return layer_group
    try:
        return schema.layer_groups[int(row["layer_group_id"])].group_id
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise AllocationValidationError("Unable to resolve layer_group from surrogate row") from exc


def _resolve_timestep_band(row: Mapping[str, Any], schema: CellSchema) -> str:
    raw_timestep_band = row.get("timestep_band")
    if isinstance(raw_timestep_band, str):
        timestep_band = raw_timestep_band.strip()
        if not timestep_band:
            raise AllocationValidationError("timestep_band must be non-empty when provided as a string")
        return timestep_band
    try:
        return schema.timestep_bands[int(row["timestep_band_id"])].band_id
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise AllocationValidationError("Unable to resolve timestep_band from surrogate row") from exc


def canonicalize_scored_row(
    row: Mapping[str, Any],
    *,
    schema: CellSchema,
) -> dict[str, Any]:
    layer_group = _resolve_layer_group(row, schema)
    timestep_band = _resolve_timestep_band(row, schema)
    cell_id = str(row.get("cell_id") or f"{layer_group}__{timestep_band}").strip()
    candidate_rank = int(row["candidate_rank"])
    if "predicted_utility" in row:
        predicted_utility = float(row["predicted_utility"])
    elif "utility_score" in row:
        predicted_utility = float(row["utility_score"])
    else:
        predicted_utility = float(row["utility"])

    expected_cell_id = f"{layer_group}__{timestep_band}"
    layer_group_ids = {group.group_id for group in schema.layer_groups}
    timestep_band_ids = {band.band_id for band in schema.timestep_bands}
    schema_cell_ids = {cell.cell_id for cell in schema.cells}

    if layer_group not in layer_group_ids:
        raise AllocationValidationError(f"Unknown layer_group {layer_group!r}")
    if timestep_band not in timestep_band_ids:
        raise AllocationValidationError(f"Unknown timestep_band {timestep_band!r}")
    if candidate_rank not in set(schema.candidate_ranks):
        raise AllocationValidationError(f"Unknown candidate_rank {candidate_rank!r}")
    if cell_id != expected_cell_id:
        raise AllocationValidationError(
            f"cell_id {cell_id!r} does not match canonical schema id {expected_cell_id!r}"
        )
    if cell_id not in schema_cell_ids:
        raise AllocationValidationError(f"Unknown cell_id {cell_id!r}")

    return {
        "cell_id": cell_id,
        "layer_group": layer_group,
        "timestep_band": timestep_band,
        "candidate_rank": candidate_rank,
        "predicted_utility": predicted_utility,
    }


def canonicalize_scored_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    schema: CellSchema | None = None,
) -> list[dict[str, Any]]:
    if not records:
        raise AllocationValidationError("At least one scored surrogate row is required")

    resolved_schema = build_allocation_schema(records, schema=schema)
    cell_order = {cell.cell_id: cell.cell_index for cell in resolved_schema.cells}
    rank_order = {rank: index for index, rank in enumerate(resolved_schema.candidate_ranks)}

    canonical_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for record in records:
        canonical = canonicalize_scored_row(record, schema=resolved_schema)
        key = (canonical["cell_id"], canonical["candidate_rank"])
        if key in seen:
            raise AllocationValidationError(f"Duplicate scored surrogate row for {key!r}")
        seen.add(key)
        canonical_rows.append(canonical)

    expected = {
        (cell.cell_id, int(candidate_rank))
        for cell in resolved_schema.cells
        for candidate_rank in resolved_schema.candidate_ranks
    }
    missing = sorted(expected - seen)
    if missing:
        preview = ", ".join(f"{cell_id}@{rank}" for cell_id, rank in missing[:5])
        raise AllocationValidationError(f"Missing canonical scored rows for {preview}")

    return sorted(
        canonical_rows,
        key=lambda item: (
            cell_order[item["cell_id"]],
            rank_order[item["candidate_rank"]],
        ),
    )


def build_choice_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    cost_field: str,
    utility_field: str,
    schema: CellSchema | None = None,
) -> list[dict[str, Any]]:
    if cost_field != "candidate_rank":
        raise AllocationValidationError("build_choice_groups only supports cost_field='candidate_rank'")
    if utility_field != "predicted_utility":
        raise AllocationValidationError("build_choice_groups only supports utility_field='predicted_utility'")

    canonical_rows = canonicalize_scored_rows(records, schema=schema)
    by_cell: dict[str, list[dict[str, Any]]] = {}
    max_rank = max(int(rank) for rank in build_allocation_schema(records, schema=schema).candidate_ranks)
    for row in canonical_rows:
        by_cell.setdefault(str(row["cell_id"]), []).append(dict(row))

    choice_groups: list[dict[str, Any]] = []
    for cell_id in sorted(by_cell):
        options = sorted(
            [
                {
                    "cell_id": str(row["cell_id"]),
                    "layer_group": str(row["layer_group"]),
                    "layer_group_id": str(row["layer_group"]),
                    "timestep_band": str(row["timestep_band"]),
                    "timestep_band_id": str(row["timestep_band"]),
                    "candidate_rank": int(row["candidate_rank"]),
                    "cost": int(row["candidate_rank"]),
                    "utility": float(row["predicted_utility"]),
                    "rank_fraction_of_max": (
                        0.0 if max_rank <= 0 else float(row["candidate_rank"]) / float(max_rank)
                    ),
                }
                for row in by_cell[cell_id]
            ],
            key=lambda item: (
                int(item["cost"]),
                -_as_float(item["utility"]),
                int(item["candidate_rank"]),
            ),
        )
        if len({int(option["candidate_rank"]) for option in options}) != len(options):
            raise AllocationValidationError(f"Duplicate candidate ranks for cell {cell_id!r}")
        choice_groups.append(
            {
                "group_id": cell_id,
                "layer_group": str(options[0]["layer_group"]),
                "timestep_band": str(options[0]["timestep_band"]),
                "options": options,
            }
        )
    if not choice_groups:
        raise AllocationValidationError("At least one allocation choice group is required")
    return choice_groups


def _state_is_better(candidate: _DpState, incumbent: _DpState | None) -> bool:
    if incumbent is None:
        return True
    if candidate.utility_scaled != incumbent.utility_scaled:
        return candidate.utility_scaled > incumbent.utility_scaled
    if candidate.used_budget != incumbent.used_budget:
        return candidate.used_budget < incumbent.used_budget
    return candidate.signature < incumbent.signature


def _build_result(
    *,
    final_state: _DpState,
    budget: int,
    utility_scale: int,
    solver_mode: str,
    used_fallback: bool,
    estimated_state_count: int,
) -> dict[str, Any]:
    selections: list[dict[str, Any]] = []
    for selection in final_state.selections:
        layer_group = str(selection.get("layer_group", selection.get("layer_group_id", ""))).strip()
        timestep_band = str(selection.get("timestep_band", selection.get("timestep_band_id", ""))).strip()
        payload = {
            "cell_id": str(selection["cell_id"]),
            "candidate_rank": int(selection["candidate_rank"]),
            "cost": int(selection["cost"]),
            "predicted_utility": _round_float(_as_float(selection["utility"])),
        }
        if layer_group:
            payload["layer_group"] = layer_group
            payload["layer_group_id"] = layer_group
        if timestep_band:
            payload["timestep_band"] = timestep_band
            payload["timestep_band_id"] = timestep_band
        if "rank_fraction_of_max" in selection:
            payload["rank_fraction_of_max"] = _round_float(_as_float(selection["rank_fraction_of_max"]))
        selections.append(payload)

    return {
        "solver_mode": solver_mode,
        "used_fallback": used_fallback,
        "budget": int(budget),
        "used_budget": int(final_state.used_budget),
        "budget_utilization": (_round_float(final_state.used_budget / float(budget)) if budget > 0 else 0.0),
        "selected_cell_count": len(selections),
        "total_predicted_utility": _round_float(final_state.utility_scaled / float(utility_scale)),
        "estimated_state_count": int(estimated_state_count),
        "selections": selections,
    }


def solve_multiple_choice_knapsack_exact(
    choice_groups: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    utility_scale: int,
    estimated_state_count: int,
) -> dict[str, Any]:
    states: dict[int, _DpState] = {
        0: _DpState(utility_scaled=0, used_budget=0, signature=tuple(), selections=tuple())
    }
    for group in choice_groups:
        options = list(group.get("options", []))
        if not options:
            raise AllocationValidationError(f"Choice group {group.get('group_id')!r} must define at least one option")
        next_states: dict[int, _DpState] = {}
        for state in states.values():
            for option in options:
                cost = int(option["cost"])
                new_budget = state.used_budget + cost
                if new_budget > budget:
                    continue
                candidate_state = _DpState(
                    utility_scaled=state.utility_scaled + int(round(_as_float(option["utility"]) * utility_scale)),
                    used_budget=new_budget,
                    signature=state.signature + (int(option["candidate_rank"]),),
                    selections=state.selections + (dict(option),),
                )
                incumbent = next_states.get(new_budget)
                if _state_is_better(candidate_state, incumbent):
                    next_states[new_budget] = candidate_state
        if not next_states:
            raise AllocationValidationError("No feasible exact allocation satisfies the requested budget")
        states = next_states

    final_state: _DpState | None = None
    for state in states.values():
        if _state_is_better(state, final_state):
            final_state = state
    if final_state is None:
        raise AllocationValidationError("Unable to construct an exact allocation result")
    return _build_result(
        final_state=final_state,
        budget=budget,
        utility_scale=utility_scale,
        solver_mode="exact_dynamic_programming",
        used_fallback=False,
        estimated_state_count=estimated_state_count,
    )


def solve_multiple_choice_knapsack_greedy(
    choice_groups: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    utility_scale: int,
    estimated_state_count: int,
) -> dict[str, Any]:
    current_indices: list[int] = []
    current_budget = 0
    current_utility = 0.0

    for group in choice_groups:
        options = list(group.get("options", []))
        if not options:
            raise AllocationValidationError(f"Choice group {group.get('group_id')!r} must define at least one option")
        baseline = min(
            options,
            key=lambda item: (
                int(item["cost"]),
                -_as_float(item["utility"]),
                int(item["candidate_rank"]),
            ),
        )
        baseline_index = options.index(baseline)
        current_indices.append(baseline_index)
        current_budget += int(baseline["cost"])
        current_utility += _as_float(baseline["utility"])

    if current_budget > budget:
        raise AllocationValidationError("Requested budget is smaller than the minimum feasible allocation cost")

    while True:
        remaining_budget = budget - current_budget
        best_upgrade: tuple[float, float, int, str, int, int] | None = None
        best_group_index = -1
        best_candidate_index = -1
        for group_index, group in enumerate(choice_groups):
            current_option = group["options"][current_indices[group_index]]
            for candidate_index, candidate_option in enumerate(group["options"]):
                extra_cost = int(candidate_option["cost"]) - int(current_option["cost"])
                if extra_cost <= 0 or extra_cost > remaining_budget:
                    continue
                gain = _as_float(candidate_option["utility"]) - _as_float(current_option["utility"])
                if gain <= 0.0:
                    continue
                candidate_key = (
                    gain / float(extra_cost),
                    gain,
                    -extra_cost,
                    str(group["group_id"]),
                    int(candidate_option["candidate_rank"]),
                    -candidate_index,
                )
                if best_upgrade is None or candidate_key > best_upgrade:
                    best_upgrade = candidate_key
                    best_group_index = group_index
                    best_candidate_index = candidate_index
        if best_upgrade is None:
            break

        group = choice_groups[best_group_index]
        current_option = group["options"][current_indices[best_group_index]]
        candidate_option = group["options"][best_candidate_index]
        current_budget += int(candidate_option["cost"]) - int(current_option["cost"])
        current_utility += _as_float(candidate_option["utility"]) - _as_float(current_option["utility"])
        current_indices[best_group_index] = best_candidate_index

    final_selections = tuple(
        dict(group["options"][current_indices[index]]) for index, group in enumerate(choice_groups)
    )
    final_state = _DpState(
        utility_scaled=int(round(current_utility * utility_scale)),
        used_budget=current_budget,
        signature=tuple(int(selection["candidate_rank"]) for selection in final_selections),
        selections=final_selections,
    )
    return _build_result(
        final_state=final_state,
        budget=budget,
        utility_scale=utility_scale,
        solver_mode="deterministic_greedy",
        used_fallback=True,
        estimated_state_count=estimated_state_count,
    )


def solve_multiple_choice_knapsack(
    choice_groups: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    utility_scale: int = 1000000,
    max_exact_state_count: int = 500000,
    fallback_mode: str = "deterministic_greedy",
) -> dict[str, Any]:
    if not choice_groups:
        raise AllocationValidationError("At least one allocation choice group is required")
    normalized_budget = _as_non_negative_int(budget, name="budget")
    estimated_state_count = max(len(choice_groups), 1) * max(normalized_budget + 1, 1)
    estimated_state_count *= max((len(group["options"]) for group in choice_groups), default=1)
    if estimated_state_count <= int(max_exact_state_count):
        return solve_multiple_choice_knapsack_exact(
            choice_groups,
            budget=normalized_budget,
            utility_scale=int(utility_scale),
            estimated_state_count=estimated_state_count,
        )
    if fallback_mode != "deterministic_greedy":
        raise AllocationValidationError(f"Unsupported fallback mode {fallback_mode!r}")
    return solve_multiple_choice_knapsack_greedy(
        choice_groups,
        budget=normalized_budget,
        utility_scale=int(utility_scale),
        estimated_state_count=estimated_state_count,
    )


def _choice_groups_for_dimension(
    canonical_rows: Sequence[Mapping[str, Any]],
    *,
    schema: CellSchema,
    group_field: str,
    group_ids: Sequence[str],
    cost_multiplier: int,
) -> list[dict[str, Any]]:
    rows_by_group_rank: dict[tuple[str, int], list[dict[str, Any]]] = {}
    max_rank = max(int(rank) for rank in schema.candidate_ranks)
    for row in canonical_rows:
        key = (str(row[group_field]), int(row["candidate_rank"]))
        rows_by_group_rank.setdefault(key, []).append(dict(row))

    choice_groups: list[dict[str, Any]] = []
    for group_id in group_ids:
        options: list[dict[str, Any]] = []
        for candidate_rank in schema.candidate_ranks:
            matching = rows_by_group_rank.get((str(group_id), int(candidate_rank)), [])
            if not matching:
                raise AllocationValidationError(
                    f"Missing canonical scored rows for {group_field}={group_id!r} rank={candidate_rank}"
                )
            options.append(
                {
                    "cell_id": str(group_id),
                    "candidate_rank": int(candidate_rank),
                    "cost": int(candidate_rank) * int(cost_multiplier),
                    "utility": _round_float(
                        sum(float(item["predicted_utility"]) for item in matching)
                    ),
                    "rank_fraction_of_max": (
                        0.0 if max_rank <= 0 else float(candidate_rank) / float(max_rank)
                    ),
                }
            )
        choice_groups.append({"group_id": str(group_id), "options": options})
    return choice_groups


def _solve_uniform(
    canonical_rows: Sequence[Mapping[str, Any]],
    *,
    schema: CellSchema,
    rank_budget_total: int,
) -> dict[str, int]:
    utility_by_rank = {int(candidate_rank): 0.0 for candidate_rank in schema.candidate_ranks}
    for row in canonical_rows:
        utility_by_rank[int(row["candidate_rank"])] += float(row["predicted_utility"])

    best_rank = 0
    best_utility = float("-inf")
    for candidate_rank in schema.candidate_ranks:
        used_budget = int(candidate_rank) * len(schema.cells)
        if used_budget > rank_budget_total:
            continue
        candidate_utility = _round_float(utility_by_rank[int(candidate_rank)])
        if candidate_utility > best_utility or (
            candidate_utility == best_utility and int(candidate_rank) < best_rank
        ):
            best_rank = int(candidate_rank)
            best_utility = candidate_utility
    return {cell.cell_id: int(best_rank) for cell in schema.cells}


def _solve_layer_only(
    canonical_rows: Sequence[Mapping[str, Any]],
    *,
    schema: CellSchema,
    rank_budget_total: int,
    utility_scale: int = 1000000,
    max_exact_state_count: int = 500000,
    fallback_mode: str = "deterministic_greedy",
) -> dict[str, int]:
    result = solve_multiple_choice_knapsack(
        _choice_groups_for_dimension(
            canonical_rows,
            schema=schema,
            group_field="layer_group",
            group_ids=[group.group_id for group in schema.layer_groups],
            cost_multiplier=len(schema.timestep_bands),
        ),
        budget=rank_budget_total,
        utility_scale=utility_scale,
        max_exact_state_count=max_exact_state_count,
        fallback_mode=fallback_mode,
    )
    rank_by_group = {str(item["cell_id"]): int(item["candidate_rank"]) for item in result["selections"]}
    return {cell.cell_id: rank_by_group[cell.layer_group.group_id] for cell in schema.cells}


def _solve_timestep_only(
    canonical_rows: Sequence[Mapping[str, Any]],
    *,
    schema: CellSchema,
    rank_budget_total: int,
    utility_scale: int = 1000000,
    max_exact_state_count: int = 500000,
    fallback_mode: str = "deterministic_greedy",
) -> dict[str, int]:
    result = solve_multiple_choice_knapsack(
        _choice_groups_for_dimension(
            canonical_rows,
            schema=schema,
            group_field="timestep_band",
            group_ids=[band.band_id for band in schema.timestep_bands],
            cost_multiplier=len(schema.layer_groups),
        ),
        budget=rank_budget_total,
        utility_scale=utility_scale,
        max_exact_state_count=max_exact_state_count,
        fallback_mode=fallback_mode,
    )
    rank_by_band = {str(item["cell_id"]): int(item["candidate_rank"]) for item in result["selections"]}
    return {cell.cell_id: rank_by_band[cell.timestep_band.band_id] for cell in schema.cells}


def _solve_proposed(
    canonical_rows: Sequence[Mapping[str, Any]],
    *,
    schema: CellSchema,
    rank_budget_total: int,
    utility_scale: int = 1000000,
    max_exact_state_count: int = 500000,
    fallback_mode: str = "deterministic_greedy",
) -> dict[str, int]:
    result = solve_multiple_choice_knapsack(
        build_choice_groups(
            canonical_rows,
            cost_field="candidate_rank",
            utility_field="predicted_utility",
            schema=schema,
        ),
        budget=rank_budget_total,
        utility_scale=utility_scale,
        max_exact_state_count=max_exact_state_count,
        fallback_mode=fallback_mode,
    )
    return {str(item["cell_id"]): int(item["candidate_rank"]) for item in result["selections"]}


def build_allocation_manifest(
    backend: str,
    *,
    schema: CellSchema,
    rank_budget_total: int,
    cell_ranks: Mapping[str, int],
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
) -> dict[str, Any]:
    if backend not in {"uniform", "layer_only", "timestep_only", "proposed"}:
        raise AllocationValidationError(f"Unsupported backend {backend!r}")

    cells: list[dict[str, Any]] = []
    for cell in schema.cells:
        if cell.cell_id not in cell_ranks:
            raise AllocationValidationError(f"Missing cell rank for {cell.cell_id!r}")
        rank = int(cell_ranks[cell.cell_id])
        adapter_name = (
            f"{backend}_bank"
            if backend in {"uniform", "layer_only"}
            else f"{backend}__{cell.timestep_band.band_id}"
        )
        cells.append(
            {
                "cell_id": cell.cell_id,
                "layer_group": cell.layer_group.group_id,
                "timestep_band": cell.timestep_band.band_id,
                "rank": rank,
                "alpha": rank,
                "target_modules": [str(module_name) for module_name in target_modules],
                "adapter_name": adapter_name,
            }
        )

    return {
        "schema_version": "1.0",
        "backend": backend,
        "rank_budget_total": int(rank_budget_total),
        "layer_groups": [group.group_id for group in schema.layer_groups],
        "timestep_bands": [band.band_id for band in schema.timestep_bands],
        "cells": cells,
    }


def validate_allocation_manifest(path: Path) -> dict[str, Any]:
    payload = _load_runtime_allocation_manifest(path)
    return {
        "ok": True,
        "selected_cell_count": len(payload["cells"]),
        "used_budget": sum(int(cell["rank"]) for cell in payload["cells"]),
    }


__all__ = [
    "CANONICAL_SCORED_ROW_FIELDNAMES",
    "AllocationValidationError",
    "build_allocation_manifest",
    "build_allocation_schema",
    "build_choice_groups",
    "canonicalize_scored_row",
    "canonicalize_scored_rows",
    "solve_multiple_choice_knapsack",
    "solve_multiple_choice_knapsack_exact",
    "solve_multiple_choice_knapsack_greedy",
    "validate_allocation_manifest",
]
