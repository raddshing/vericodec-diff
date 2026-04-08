from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rd_lora.probe import DEFAULT_ACCELERATE_CONFIG, run_preflight
from rd_lora.substrate.diffusers_sdxl import deep_update, display_path, resolve_path, save_json
from rd_lora.surrogate import (
    DEFAULT_SURROGATE_FEATURE_NAMES,
    LinearUtilitySurrogate,
    fit_linear_surrogate,
    load_utility_records,
    score_records,
    write_metric_csv,
    write_prediction_csv,
)


DEFAULT_ALLOCATOR_CONFIG_PATH = "configs/rdlora_allocator.yaml"
DEFAULT_ALLOCATOR_OUTPUT_ROOT = "outputs/rd_lora/allocator"
ALLOCATION_CHOICE_FIELDNAMES = (
    "cell_id",
    "layer_group_id",
    "timestep_band_id",
    "candidate_rank",
    "cost",
    "predicted_utility",
    "rank_fraction_of_max",
    "is_attention_cell",
    "layer_count",
    "step_count",
    "event_count",
)
ALLOCATION_SUMMARY_FIELDNAMES = (
    "budget",
    "used_budget",
    "budget_utilization",
    "selected_cell_count",
    "total_predicted_utility",
    "solver_mode",
    "used_fallback",
)


class AllocationValidationError(ValueError):
    """Raised when the allocator config or outputs are invalid."""


@dataclass(frozen=True)
class _DpState:
    utility_scaled: int
    used_budget: int
    signature: tuple[int, ...]
    selections: tuple[dict[str, Any], ...]


def default_allocator_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "probe_run_dir": "outputs/rd_lora/probe/stageb_cpu_smoke",
            "output_root": DEFAULT_ALLOCATOR_OUTPUT_ROOT,
            "accelerate_config": DEFAULT_ACCELERATE_CONFIG,
        },
        "run": {
            "name": "stagec_smoke",
            "seed": 20260409,
        },
        "preflight": {
            "require_torch_cuda": True,
            "required_diffusers_version": "0.38.0.dev0",
            "required_accelerate_config": DEFAULT_ACCELERATE_CONFIG,
        },
        "surrogate": {
            "feature_names": list(DEFAULT_SURROGATE_FEATURE_NAMES),
            "l2_regularization": 1.0e-6,
            "clip_min_utility": 0.0,
            "round_digits": 6,
        },
        "allocation": {
            "budget": 96,
            "cost_field": "candidate_rank",
            "utility_field": "predicted_utility",
            "utility_scale": 1000000,
            "max_exact_state_count": 500000,
            "fallback_mode": "deterministic_greedy",
        },
    }


def _validate_repo_local_path(repo_root: Path, path: Path, *, name: str) -> None:
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise AllocationValidationError(f"{name} must resolve inside repo_root for deterministic paths") from exc


def _validate_single_path_token(raw_value: str, *, name: str) -> str:
    token = raw_value.strip()
    if not token:
        raise AllocationValidationError(f"{name} must be a non-empty path token")
    token_path = Path(token)
    if len(token_path.parts) != 1 or token_path.name != token or token in {".", ".."}:
        raise AllocationValidationError(f"{name} must be a single path component")
    return token


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fieldnames})


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


def resolve_allocator_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_allocator_config(), raw_config)
    paths = dict(config.get("paths", {}))
    run = dict(config.get("run", {}))
    preflight = dict(config.get("preflight", {}))
    surrogate = dict(config.get("surrogate", {}))
    allocation = dict(config.get("allocation", {}))

    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()
    probe_run_dir = resolve_path(resolved_repo_root, str(paths.get("probe_run_dir"))).resolve()
    output_root = resolve_path(
        resolved_repo_root,
        str(paths.get("output_root", DEFAULT_ALLOCATOR_OUTPUT_ROOT)),
    ).resolve()
    accelerate_config = resolve_path(
        resolved_repo_root,
        str(paths.get("accelerate_config", DEFAULT_ACCELERATE_CONFIG)),
    ).resolve()
    required_accelerate_config = resolve_path(
        resolved_repo_root,
        str(preflight.get("required_accelerate_config", DEFAULT_ACCELERATE_CONFIG)),
    ).resolve()

    run_name = _validate_single_path_token(str(run.get("name", "stagec_smoke")), name="run.name")
    run_dir = output_root / run_name
    surrogate_dir = run_dir / "surrogate"
    allocation_dir = run_dir / "allocation"
    utility_records_json = probe_run_dir / "utility_records.json"
    probe_summary_json = probe_run_dir / "probe_summary.json"

    path_map = {
        "paths.probe_run_dir": probe_run_dir,
        "paths.output_root": output_root,
        "derived run_dir": run_dir,
        "derived surrogate_dir": surrogate_dir,
        "derived allocation_dir": allocation_dir,
        "paths.accelerate_config": accelerate_config,
        "preflight.required_accelerate_config": required_accelerate_config,
        "derived utility_records_json": utility_records_json,
        "derived probe_summary_json": probe_summary_json,
    }
    for name, path in path_map.items():
        _validate_repo_local_path(resolved_repo_root, path, name=name)

    feature_names = [str(value) for value in surrogate.get("feature_names", DEFAULT_SURROGATE_FEATURE_NAMES)]
    if not feature_names:
        raise AllocationValidationError("surrogate.feature_names must not be empty")
    l2_regularization = float(surrogate.get("l2_regularization", 1.0e-6))
    if l2_regularization < 0.0:
        raise AllocationValidationError("surrogate.l2_regularization must be non-negative")
    round_digits = int(surrogate.get("round_digits", 6))
    if round_digits < 0:
        raise AllocationValidationError("surrogate.round_digits must be non-negative")

    budget = _as_non_negative_int(allocation.get("budget", 0), name="allocation.budget")
    utility_scale = _as_non_negative_int(
        allocation.get("utility_scale", 1000000),
        name="allocation.utility_scale",
    )
    if utility_scale == 0:
        raise AllocationValidationError("allocation.utility_scale must be positive")
    max_exact_state_count = _as_non_negative_int(
        allocation.get("max_exact_state_count", 500000),
        name="allocation.max_exact_state_count",
    )
    if max_exact_state_count == 0:
        raise AllocationValidationError("allocation.max_exact_state_count must be positive")
    fallback_mode = str(allocation.get("fallback_mode", "deterministic_greedy")).strip()
    if fallback_mode not in {"deterministic_greedy"}:
        raise AllocationValidationError("allocation.fallback_mode must be 'deterministic_greedy'")

    return {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "probe_run_dir": str(probe_run_dir),
            "probe_summary_json": str(probe_summary_json),
            "utility_records_json": str(utility_records_json),
            "output_root": str(output_root),
            "run_dir": str(run_dir),
            "surrogate_dir": str(surrogate_dir),
            "surrogate_model_json": str(surrogate_dir / "surrogate_model.json"),
            "surrogate_predictions_json": str(surrogate_dir / "surrogate_predictions.json"),
            "surrogate_predictions_csv": str(surrogate_dir / "surrogate_predictions.csv"),
            "surrogate_metrics_csv": str(surrogate_dir / "surrogate_metrics.csv"),
            "surrogate_summary_json": str(surrogate_dir / "surrogate_training_summary.json"),
            "allocation_dir": str(allocation_dir),
            "allocation_manifest_json": str(allocation_dir / "allocation_manifest.json"),
            "allocation_choices_csv": str(allocation_dir / "allocation_choices.csv"),
            "allocation_summary_csv": str(allocation_dir / "allocation_summary.csv"),
            "accelerate_config": str(accelerate_config),
        },
        "run": {
            "name": run_name,
            "seed": int(run.get("seed", 20260409)),
        },
        "preflight": {
            "require_torch_cuda": bool(preflight.get("require_torch_cuda", True)),
            "required_diffusers_version": preflight.get("required_diffusers_version"),
            "required_accelerate_config": str(required_accelerate_config),
        },
        "surrogate": {
            "feature_names": feature_names,
            "l2_regularization": l2_regularization,
            "clip_min_utility": surrogate.get("clip_min_utility"),
            "round_digits": round_digits,
        },
        "allocation": {
            "budget": budget,
            "cost_field": str(allocation.get("cost_field", "candidate_rank")),
            "utility_field": str(allocation.get("utility_field", "predicted_utility")),
            "utility_scale": utility_scale,
            "max_exact_state_count": max_exact_state_count,
            "fallback_mode": fallback_mode,
        },
    }


def train_surrogate_from_config(
    config: Mapping[str, Any],
    *,
    resolved_config_path: Path,
) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    utility_records_path = Path(config["paths"]["utility_records_json"])
    if not utility_records_path.is_file():
        raise AllocationValidationError(f"Missing probe utility records JSON: {utility_records_path}")

    surrogate_dir = Path(config["paths"]["surrogate_dir"])
    surrogate_dir.mkdir(parents=True, exist_ok=True)
    preflight = run_preflight(config)

    utility_records = load_utility_records(utility_records_path)
    fit_result = fit_linear_surrogate(
        utility_records,
        feature_names=config["surrogate"]["feature_names"],
        l2_regularization=float(config["surrogate"]["l2_regularization"]),
        clip_min_utility=config["surrogate"]["clip_min_utility"],
        round_digits=int(config["surrogate"]["round_digits"]),
    )
    model = fit_result["model"]
    scored_records = fit_result["records"]
    metrics = fit_result["metrics"]

    model_json_path = Path(config["paths"]["surrogate_model_json"])
    predictions_json_path = Path(config["paths"]["surrogate_predictions_json"])
    predictions_csv_path = Path(config["paths"]["surrogate_predictions_csv"])
    metrics_csv_path = Path(config["paths"]["surrogate_metrics_csv"])
    summary_json_path = Path(config["paths"]["surrogate_summary_json"])

    save_json(
        model_json_path,
        {
            **model.to_dict(),
            "resolved_config": display_path(resolved_config_path, repo_root),
            "source_probe_summary": display_path(Path(config["paths"]["probe_summary_json"]), repo_root),
            "source_utility_records": display_path(utility_records_path, repo_root),
            "record_count": len(scored_records),
            "metrics": metrics,
        },
    )
    save_json(
        predictions_json_path,
        {
            "schema_version": 1,
            "resolved_config": display_path(resolved_config_path, repo_root),
            "source_utility_records": display_path(utility_records_path, repo_root),
            "feature_names": list(model.feature_names),
            "record_count": len(scored_records),
            "records": scored_records,
        },
    )
    write_prediction_csv(predictions_csv_path, scored_records)
    write_metric_csv(metrics_csv_path, metrics)

    summary_payload = {
        "schema_version": 1,
        "resolved_config": display_path(resolved_config_path, repo_root),
        "source_probe_summary": display_path(Path(config["paths"]["probe_summary_json"]), repo_root),
        "source_utility_records": display_path(utility_records_path, repo_root),
        "model_json": display_path(model_json_path, repo_root),
        "predictions_json": display_path(predictions_json_path, repo_root),
        "predictions_csv": display_path(predictions_csv_path, repo_root),
        "metrics_csv": display_path(metrics_csv_path, repo_root),
        "record_count": len(scored_records),
        "cell_count": len({str(record["cell_id"]) for record in scored_records}),
        "preflight": preflight,
        "metrics": metrics,
        "completed_successfully": True,
    }
    save_json(summary_json_path, summary_payload)
    summary_payload["summary_json"] = display_path(summary_json_path, repo_root)
    return summary_payload


def load_surrogate_predictions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AllocationValidationError(f"{path} must parse to a mapping")
    records = payload.get("records")
    if not isinstance(records, list):
        raise AllocationValidationError(f"{path} must define a records list")
    normalized: list[dict[str, Any]] = []
    for record in records:
        payload_record = dict(record)
        if "predicted_utility" not in payload_record:
            raise AllocationValidationError(f"{path} records must include predicted_utility")
        normalized.append(payload_record)
    return sorted(
        normalized,
        key=lambda item: (
            str(item["cell_id"]),
            int(item["candidate_rank"]),
            str(item["layer_group_id"]),
            str(item["timestep_band_id"]),
        ),
    )


def build_choice_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    cost_field: str,
    utility_field: str,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        cell_id = str(record["cell_id"])
        group = groups.setdefault(
            cell_id,
            {
                "group_id": cell_id,
                "layer_group_id": str(record["layer_group_id"]),
                "timestep_band_id": str(record["timestep_band_id"]),
                "is_attention_cell": bool(record["is_attention_cell"]),
                "options": [],
            },
        )
        option = dict(record)
        option["cost"] = _as_non_negative_int(record[cost_field], name=f"{cell_id}.{cost_field}")
        option["utility"] = _as_float(record[utility_field])
        group["options"].append(option)

    if not groups:
        raise AllocationValidationError("At least one allocation group is required")

    choice_groups: list[dict[str, Any]] = []
    for group_id in sorted(groups):
        group = groups[group_id]
        options = sorted(
            group["options"],
            key=lambda item: (
                int(item["cost"]),
                -_as_float(item["utility"]),
                int(item["candidate_rank"]),
            ),
        )
        candidate_ranks = [int(item["candidate_rank"]) for item in options]
        if len(candidate_ranks) != len(set(candidate_ranks)):
            raise AllocationValidationError(f"Allocation group {group_id!r} contains duplicate candidate ranks")
        choice_groups.append(
            {
                **group,
                "options": options,
            }
        )
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
    choice_groups: Sequence[Mapping[str, Any]],
    final_state: _DpState,
    budget: int,
    utility_scale: int,
    solver_mode: str,
    used_fallback: bool,
    estimated_state_count: int,
) -> dict[str, Any]:
    selections: list[dict[str, Any]] = []
    for group, selection in zip(choice_groups, final_state.selections):
        selections.append(
            {
                "cell_id": str(selection["cell_id"]),
                "layer_group_id": str(selection["layer_group_id"]),
                "timestep_band_id": str(selection["timestep_band_id"]),
                "candidate_rank": int(selection["candidate_rank"]),
                "cost": int(selection["cost"]),
                "predicted_utility": _round_float(_as_float(selection["utility"])),
                "rank_fraction_of_max": _round_float(_as_float(selection["rank_fraction_of_max"])),
                "is_attention_cell": bool(group["is_attention_cell"]),
                "layer_count": int(selection["layer_count"]),
                "step_count": int(selection["step_count"]),
                "event_count": int(selection["event_count"]),
            }
        )
    total_predicted_utility = _round_float(final_state.utility_scaled / float(utility_scale))
    return {
        "solver_mode": solver_mode,
        "used_fallback": used_fallback,
        "budget": int(budget),
        "used_budget": int(final_state.used_budget),
        "budget_utilization": (_round_float(final_state.used_budget / float(budget)) if budget > 0 else 0.0),
        "selected_cell_count": len(selections),
        "total_predicted_utility": total_predicted_utility,
        "estimated_state_count": estimated_state_count,
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
        next_states: dict[int, _DpState] = {}
        for state in states.values():
            for option in group["options"]:
                cost = int(option["cost"])
                new_budget = state.used_budget + cost
                if new_budget > budget:
                    continue
                utility_scaled = int(round(_as_float(option["utility"]) * utility_scale))
                candidate_state = _DpState(
                    utility_scaled=state.utility_scaled + utility_scaled,
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
        choice_groups=choice_groups,
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
        baseline = min(
            group["options"],
            key=lambda item: (
                int(item["cost"]),
                -_as_float(item["utility"]),
                int(item["candidate_rank"]),
            ),
        )
        baseline_index = group["options"].index(baseline)
        current_indices.append(baseline_index)
        current_budget += int(baseline["cost"])
        current_utility += _as_float(baseline["utility"])
    if current_budget > budget:
        raise AllocationValidationError("Requested budget is smaller than the minimum feasible allocation cost")

    while True:
        remaining_budget = budget - current_budget
        best_upgrade: tuple[float, float, int, str, int, int] | None = None
        for group_index, group in enumerate(choice_groups):
            current_option = group["options"][current_indices[group_index]]
            for candidate_index, candidate_option in enumerate(group["options"]):
                extra_cost = int(candidate_option["cost"]) - int(current_option["cost"])
                if extra_cost <= 0 or extra_cost > remaining_budget:
                    continue
                gain = _as_float(candidate_option["utility"]) - _as_float(current_option["utility"])
                if gain <= 0.0:
                    continue
                ratio = gain / float(extra_cost)
                candidate_key = (
                    ratio,
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
        choice_groups=choice_groups,
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


def solve_allocation_from_config(
    config: Mapping[str, Any],
    *,
    resolved_config_path: Path,
) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    preflight = run_preflight(config)
    predictions_json_path = Path(config["paths"]["surrogate_predictions_json"])
    if not predictions_json_path.is_file():
        model_json_path = Path(config["paths"]["surrogate_model_json"])
        utility_records_json = Path(config["paths"]["utility_records_json"])
        if not utility_records_json.is_file():
            raise AllocationValidationError(
                f"Missing surrogate predictions {predictions_json_path} and utility records {utility_records_json}"
            )
        if not model_json_path.is_file():
            raise AllocationValidationError(f"Missing surrogate model JSON: {model_json_path}")
        model_payload = json.loads(model_json_path.read_text(encoding="utf-8"))
        model = LinearUtilitySurrogate.from_dict(model_payload)
        utility_records = load_utility_records(utility_records_json)
        scored_records = score_records(model, utility_records)
        save_json(
            predictions_json_path,
            {
                "schema_version": 1,
                "resolved_config": display_path(resolved_config_path, repo_root),
                "source_utility_records": display_path(utility_records_json, repo_root),
                "feature_names": list(model.feature_names),
                "record_count": len(scored_records),
                "records": scored_records,
            },
        )
        write_prediction_csv(Path(config["paths"]["surrogate_predictions_csv"]), scored_records)

    prediction_records = load_surrogate_predictions(predictions_json_path)
    choice_groups = build_choice_groups(
        prediction_records,
        cost_field=str(config["allocation"]["cost_field"]),
        utility_field=str(config["allocation"]["utility_field"]),
    )
    result = solve_multiple_choice_knapsack(
        choice_groups,
        budget=int(config["allocation"]["budget"]),
        utility_scale=int(config["allocation"]["utility_scale"]),
        max_exact_state_count=int(config["allocation"]["max_exact_state_count"]),
        fallback_mode=str(config["allocation"]["fallback_mode"]),
    )

    allocation_dir = Path(config["paths"]["allocation_dir"])
    allocation_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config["paths"]["allocation_manifest_json"])
    choices_csv_path = Path(config["paths"]["allocation_choices_csv"])
    summary_csv_path = Path(config["paths"]["allocation_summary_csv"])

    manifest = {
        "schema_version": 1,
        "resolved_config": display_path(resolved_config_path, repo_root),
        "source_probe_run_dir": display_path(Path(config["paths"]["probe_run_dir"]), repo_root),
        "source_surrogate_predictions": display_path(predictions_json_path, repo_root),
        "preflight": preflight,
        "budget": {
            "available": int(result["budget"]),
            "used": int(result["used_budget"]),
            "utilization": result["budget_utilization"],
        },
        "solver": {
            "mode": result["solver_mode"],
            "used_fallback": bool(result["used_fallback"]),
            "estimated_state_count": int(result["estimated_state_count"]),
        },
        "totals": {
            "selected_cell_count": int(result["selected_cell_count"]),
            "total_predicted_utility": result["total_predicted_utility"],
        },
        "selections": result["selections"],
    }
    save_json(manifest_path, manifest)
    _write_csv(choices_csv_path, result["selections"], fieldnames=ALLOCATION_CHOICE_FIELDNAMES)
    _write_csv(
        summary_csv_path,
        [
            {
                "budget": int(result["budget"]),
                "used_budget": int(result["used_budget"]),
                "budget_utilization": result["budget_utilization"],
                "selected_cell_count": int(result["selected_cell_count"]),
                "total_predicted_utility": result["total_predicted_utility"],
                "solver_mode": result["solver_mode"],
                "used_fallback": int(bool(result["used_fallback"])),
            }
        ],
        fieldnames=ALLOCATION_SUMMARY_FIELDNAMES,
    )
    validate_allocation_manifest(manifest_path)
    return {
        "manifest_path": display_path(manifest_path, repo_root),
        "choices_csv_path": display_path(choices_csv_path, repo_root),
        "summary_csv_path": display_path(summary_csv_path, repo_root),
        "budget": manifest["budget"],
        "solver": manifest["solver"],
        "totals": manifest["totals"],
        "selections": manifest["selections"],
        "preflight": preflight,
    }


def validate_allocation_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AllocationValidationError(f"{path} must parse to a mapping")
    required_keys = {
        "schema_version",
        "resolved_config",
        "source_probe_run_dir",
        "source_surrogate_predictions",
        "preflight",
        "budget",
        "solver",
        "totals",
        "selections",
    }
    missing = sorted(required_keys - set(payload))
    if missing:
        raise AllocationValidationError(f"{path} is missing keys: {missing}")
    selections = payload["selections"]
    if not isinstance(selections, list) or not selections:
        raise AllocationValidationError("allocation manifest must contain a non-empty selections list")
    used_budget = sum(int(selection["cost"]) for selection in selections)
    total_utility = _round_float(sum(_as_float(selection["predicted_utility"]) for selection in selections))
    if used_budget != int(payload["budget"]["used"]):
        raise AllocationValidationError("allocation manifest budget.used does not match the selected costs")
    if used_budget > int(payload["budget"]["available"]):
        raise AllocationValidationError("allocation manifest exceeds the available budget")
    if total_utility != _round_float(_as_float(payload["totals"]["total_predicted_utility"])):
        raise AllocationValidationError("allocation manifest total_predicted_utility does not match selections")
    return {
        "ok": True,
        "selected_cell_count": len(selections),
        "used_budget": used_budget,
        "total_predicted_utility": total_utility,
    }


__all__ = [
    "ALLOCATION_CHOICE_FIELDNAMES",
    "ALLOCATION_SUMMARY_FIELDNAMES",
    "DEFAULT_ALLOCATOR_CONFIG_PATH",
    "AllocationValidationError",
    "build_choice_groups",
    "default_allocator_config",
    "resolve_allocator_config",
    "solve_allocation_from_config",
    "solve_multiple_choice_knapsack",
    "solve_multiple_choice_knapsack_exact",
    "train_surrogate_from_config",
    "validate_allocation_manifest",
]
