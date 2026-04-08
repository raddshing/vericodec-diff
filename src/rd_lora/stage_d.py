from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rd_lora.allocator import build_choice_groups, solve_multiple_choice_knapsack
from rd_lora.probe import DEFAULT_ACCELERATE_CONFIG
from rd_lora.substrate.diffusers_sdxl import deep_update, display_path, resolve_path, save_json
from rd_lora.surrogate import fit_linear_surrogate, load_utility_records, score_records


DEFAULT_STAGE_D_CONFIG_PATH = "configs/rdlora_stage_d.yaml"
DEFAULT_STAGE_D_OUTPUT_ROOT = "outputs/rd_lora/stage_d"
DEFAULT_STAGE_D_MEMO_ROOT = "outputs/memos"
DEFAULT_LOCKED_METRIC_NAME = "measured_utility_mass"
METHOD_ORDER = ("uniform", "layer_only", "timestep_only", "proposed", "t_lora")
EVAL_SUMMARY_FILENAME = "evaluation_summary.json"
BASELINE_TABLE_FILENAME = "baseline_comparison.csv"
TRAINING_SUMMARY_FILENAME = "training_summary.json"


class StageDValidationError(ValueError):
    """Raised when Stage D pilot configuration or artifacts are malformed."""


def _round_float(value: float, *, digits: int = 6) -> float:
    return round(float(value), int(digits))


def _validate_repo_local_path(repo_root: Path, path: Path, *, name: str) -> None:
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise StageDValidationError(f"{name} must resolve inside repo_root for deterministic paths") from exc


def _validate_single_path_token(raw_value: str, *, name: str) -> str:
    token = raw_value.strip()
    if not token:
        raise StageDValidationError(f"{name} must be a non-empty path token")
    token_path = Path(token)
    if len(token_path.parts) != 1 or token_path.name != token or token in {".", ".."}:
        raise StageDValidationError(f"{name} must be a single path component")
    return token


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StageDValidationError(f"{path} must parse to a mapping")
    return payload


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def default_stage_d_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "tasks_config": "configs/rdlora_tasks.yaml",
            "probe_run_dir": "outputs/rd_lora/probe/week2_probe",
            "allocator_run_dir": "outputs/rd_lora/allocator/week2_allocator",
            "vanilla_config": "configs/rdlora_vanilla.yaml",
            "tlora_config": "configs/tlora_baseline.yaml",
            "output_root": DEFAULT_STAGE_D_OUTPUT_ROOT,
            "memo_output_dir": DEFAULT_STAGE_D_MEMO_ROOT,
            "accelerate_config": DEFAULT_ACCELERATE_CONFIG,
        },
        "run": {
            "name": "week2_gate",
            "seed": 20260409,
        },
        "preflight": {
            "require_torch_cuda": True,
            "required_diffusers_version": "0.38.0.dev0",
            "required_accelerate_config": DEFAULT_ACCELERATE_CONFIG,
            "required_gpu_index": 0,
            "required_gpu_name_substring": "RTX 3090",
            "required_gpu_memory_gib_min": 24,
        },
        "pilot": {
            "task_ids": [
                "t_lora_dog_subject",
                "synthetic_signage_domain",
            ],
        },
        "methods": {
            "enabled": list(METHOD_ORDER),
        },
        "allocation": {
            "budget": 96,
            "utility_field": "utility_score",
            "prediction_field": "predicted_utility",
            "top_mass_fraction": 0.20,
            "holdout_fraction": 0.25,
            "top_k_cells": 5,
        },
        "training": {
            "execute": False,
            "gradient_checkpointing": True,
            "mixed_precision": "fp16",
            "allow_missing_backends": True,
        },
        "evaluation": {
            "locked_metric_name": DEFAULT_LOCKED_METRIC_NAME,
            "locked_metric_kind": "allocation_proxy",
        },
        "gate": {
            "thresholds": {
                "top_20_mass_min": 0.60,
                "held_out_spearman_min": 0.50,
                "top_5_precision_min": 0.60,
                "proposed_relative_gain_min": 0.10,
                "overhead_ratio_max": 1.5,
            }
        },
    }


def resolve_stage_d_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_stage_d_config(), raw_config)
    paths = dict(config.get("paths", {}))
    run = dict(config.get("run", {}))
    preflight = dict(config.get("preflight", {}))
    pilot = dict(config.get("pilot", {}))
    methods = dict(config.get("methods", {}))
    allocation = dict(config.get("allocation", {}))
    training = dict(config.get("training", {}))
    evaluation = dict(config.get("evaluation", {}))
    gate = dict(config.get("gate", {}))
    thresholds = dict(gate.get("thresholds", {}))

    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()
    tasks_config = resolve_path(resolved_repo_root, str(paths.get("tasks_config", "configs/rdlora_tasks.yaml"))).resolve()
    probe_run_dir = resolve_path(resolved_repo_root, str(paths.get("probe_run_dir"))).resolve()
    allocator_run_dir = resolve_path(resolved_repo_root, str(paths.get("allocator_run_dir"))).resolve()
    vanilla_config = resolve_path(resolved_repo_root, str(paths.get("vanilla_config"))).resolve()
    tlora_config = resolve_path(resolved_repo_root, str(paths.get("tlora_config"))).resolve()
    output_root = resolve_path(resolved_repo_root, str(paths.get("output_root", DEFAULT_STAGE_D_OUTPUT_ROOT))).resolve()
    memo_output_dir = resolve_path(
        resolved_repo_root,
        str(paths.get("memo_output_dir", DEFAULT_STAGE_D_MEMO_ROOT)),
    ).resolve()
    accelerate_config = resolve_path(
        resolved_repo_root,
        str(paths.get("accelerate_config", DEFAULT_ACCELERATE_CONFIG)),
    ).resolve()
    required_accelerate_config = resolve_path(
        resolved_repo_root,
        str(preflight.get("required_accelerate_config", DEFAULT_ACCELERATE_CONFIG)),
    ).resolve()

    for required_file in (tasks_config, vanilla_config, tlora_config, accelerate_config, required_accelerate_config):
        if not required_file.exists():
            raise FileNotFoundError(f"Required Stage D path not found: {required_file}")

    for required_dir in (probe_run_dir, allocator_run_dir):
        if not required_dir.exists():
            raise FileNotFoundError(f"Required Stage D artifact directory not found: {required_dir}")

    for name, path in {
        "paths.output_root": output_root,
        "paths.memo_output_dir": memo_output_dir,
        "paths.accelerate_config": accelerate_config,
        "preflight.required_accelerate_config": required_accelerate_config,
    }.items():
        _validate_repo_local_path(resolved_repo_root, path, name=name)

    run_name = _validate_single_path_token(str(run.get("name", "week2_gate")), name="run.name")
    run_dir = (output_root / run_name).resolve()
    training_dir = (run_dir / "training").resolve()
    evaluation_dir = (run_dir / "evaluation").resolve()
    _validate_repo_local_path(resolved_repo_root, run_dir, name="derived run_dir")

    task_ids = [str(value).strip() for value in pilot.get("task_ids", []) if str(value).strip()]
    if not task_ids:
        raise StageDValidationError("pilot.task_ids must contain at least one task id")

    enabled_methods = [str(value).strip() for value in methods.get("enabled", []) if str(value).strip()]
    if not enabled_methods:
        raise StageDValidationError("methods.enabled must contain at least one method")
    unsupported_methods = [method for method in enabled_methods if method not in METHOD_ORDER]
    if unsupported_methods:
        raise StageDValidationError(f"Unsupported Stage D methods: {unsupported_methods}")

    budget = int(allocation.get("budget", 96))
    if budget <= 0:
        raise StageDValidationError("allocation.budget must be positive")

    top_mass_fraction = float(allocation.get("top_mass_fraction", 0.20))
    holdout_fraction = float(allocation.get("holdout_fraction", 0.25))
    if top_mass_fraction <= 0.0 or top_mass_fraction > 1.0:
        raise StageDValidationError("allocation.top_mass_fraction must lie within (0, 1]")
    if holdout_fraction <= 0.0 or holdout_fraction >= 1.0:
        raise StageDValidationError("allocation.holdout_fraction must lie within (0, 1)")

    top_k_cells = int(allocation.get("top_k_cells", 5))
    if top_k_cells <= 0:
        raise StageDValidationError("allocation.top_k_cells must be positive")

    locked_metric_name = str(evaluation.get("locked_metric_name", DEFAULT_LOCKED_METRIC_NAME)).strip()
    if locked_metric_name != DEFAULT_LOCKED_METRIC_NAME:
        raise StageDValidationError(
            f"evaluation.locked_metric_name must remain {DEFAULT_LOCKED_METRIC_NAME!r} for the week-2 gate"
        )

    return {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "tasks_config": str(tasks_config),
            "probe_run_dir": str(probe_run_dir),
            "allocator_run_dir": str(allocator_run_dir),
            "vanilla_config": str(vanilla_config),
            "tlora_config": str(tlora_config),
            "output_root": str(output_root),
            "memo_output_dir": str(memo_output_dir),
            "accelerate_config": str(accelerate_config),
            "required_accelerate_config": str(required_accelerate_config),
            "run_dir": str(run_dir),
            "training_dir": str(training_dir),
            "evaluation_dir": str(evaluation_dir),
            "evaluation_summary_json": str(evaluation_dir / EVAL_SUMMARY_FILENAME),
            "baseline_table_csv": str(evaluation_dir / BASELINE_TABLE_FILENAME),
            "training_summary_json": str(training_dir / TRAINING_SUMMARY_FILENAME),
        },
        "run": {
            "name": run_name,
            "seed": int(run.get("seed", 20260409)),
        },
        "preflight": {
            "require_torch_cuda": bool(preflight.get("require_torch_cuda", True)),
            "required_diffusers_version": preflight.get("required_diffusers_version"),
            "required_gpu_index": int(preflight.get("required_gpu_index", 0)),
            "required_gpu_name_substring": str(preflight.get("required_gpu_name_substring", "")).strip(),
            "required_gpu_memory_gib_min": int(preflight.get("required_gpu_memory_gib_min", 24)),
        },
        "pilot": {
            "task_ids": task_ids,
        },
        "methods": {
            "enabled": enabled_methods,
        },
        "allocation": {
            "budget": budget,
            "utility_field": str(allocation.get("utility_field", "utility_score")).strip(),
            "prediction_field": str(allocation.get("prediction_field", "predicted_utility")).strip(),
            "top_mass_fraction": top_mass_fraction,
            "holdout_fraction": holdout_fraction,
            "top_k_cells": top_k_cells,
        },
        "training": {
            "execute": bool(training.get("execute", False)),
            "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
            "mixed_precision": str(training.get("mixed_precision", "fp16")).strip(),
            "allow_missing_backends": bool(training.get("allow_missing_backends", True)),
        },
        "evaluation": {
            "locked_metric_name": locked_metric_name,
            "locked_metric_kind": str(evaluation.get("locked_metric_kind", "allocation_proxy")).strip(),
        },
        "gate": {
            "thresholds": {
                "top_20_mass_min": float(thresholds.get("top_20_mass_min", 0.60)),
                "held_out_spearman_min": float(thresholds.get("held_out_spearman_min", 0.50)),
                "top_5_precision_min": float(thresholds.get("top_5_precision_min", 0.60)),
                "proposed_relative_gain_min": float(thresholds.get("proposed_relative_gain_min", 0.10)),
                "overhead_ratio_max": float(thresholds.get("overhead_ratio_max", 1.5)),
            }
        },
    }


def collect_preflight_report(config: Mapping[str, Any]) -> dict[str, Any]:
    required_gpu_index = int(config["preflight"]["required_gpu_index"])
    required_gpu_name_substring = str(config["preflight"]["required_gpu_name_substring"])
    required_gpu_memory_gib_min = int(config["preflight"]["required_gpu_memory_gib_min"])
    accelerate_config = Path(config["paths"]["accelerate_config"]).resolve()
    required_accelerate_config = Path(config["paths"]["required_accelerate_config"]).resolve()
    report = {
        "ok": True,
        "issues": [],
        "accelerate_config": str(accelerate_config),
        "required_accelerate_config": str(required_accelerate_config),
        "accelerate_config_matches": accelerate_config == required_accelerate_config,
        "torch_cuda_is_available": None,
        "torch_device_count": None,
        "torch_device_name": None,
        "torch_device_total_memory_gib": None,
        "diffusers_version": None,
        "nvidia_smi_gpu_rows": [],
    }
    if not report["accelerate_config_matches"]:
        report["issues"].append("paths.accelerate_config does not match the required repo-local accelerate config.")

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        completed = None

    if completed is not None and completed.returncode == 0:
        gpu_rows: list[dict[str, Any]] = []
        for raw_line in completed.stdout.splitlines():
            parts = [item.strip() for item in raw_line.split(",")]
            if len(parts) != 3:
                continue
            try:
                gpu_rows.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mib": int(parts[2]),
                    }
                )
            except ValueError:
                continue
        report["nvidia_smi_gpu_rows"] = gpu_rows

    if bool(config["preflight"]["require_torch_cuda"]):
        try:
            import torch
        except Exception as exc:
            report["issues"].append(f"Unable to import torch: {type(exc).__name__}: {exc}")
        else:
            report["torch_cuda_is_available"] = bool(torch.cuda.is_available())
            report["torch_device_count"] = int(torch.cuda.device_count())
            if report["torch_cuda_is_available"] and report["torch_device_count"] > required_gpu_index:
                properties = torch.cuda.get_device_properties(required_gpu_index)
                report["torch_device_name"] = torch.cuda.get_device_name(required_gpu_index)
                report["torch_device_total_memory_gib"] = _round_float(
                    float(properties.total_memory) / float(1024**3),
                    digits=3,
                )
            if not report["torch_cuda_is_available"]:
                report["issues"].append("torch.cuda.is_available() must be True before Stage D runs.")
            elif report["torch_device_count"] <= required_gpu_index:
                report["issues"].append(f"Required GPU index {required_gpu_index} is not visible to torch.")
            else:
                if required_gpu_name_substring and required_gpu_name_substring not in str(report["torch_device_name"]):
                    report["issues"].append(
                        "GPU 0 name mismatch: "
                        f"expected substring {required_gpu_name_substring!r}, found {report['torch_device_name']!r}."
                    )
                if float(report["torch_device_total_memory_gib"] or 0.0) < float(required_gpu_memory_gib_min):
                    report["issues"].append(
                        "GPU 0 memory mismatch: "
                        f"expected at least {required_gpu_memory_gib_min} GiB, "
                        f"found {report['torch_device_total_memory_gib']}."
                    )

    required_diffusers_version = config["preflight"]["required_diffusers_version"]
    if required_diffusers_version not in (None, ""):
        try:
            report["diffusers_version"] = importlib.metadata.version("diffusers")
        except importlib.metadata.PackageNotFoundError:
            report["issues"].append("diffusers is not installed in the active environment.")
        else:
            if report["diffusers_version"] != str(required_diffusers_version):
                report["issues"].append(
                    f"diffusers version must be {required_diffusers_version}, found {report['diffusers_version']}."
                )

    report["ok"] = not report["issues"]
    return report


def ensure_preflight_ok(report: Mapping[str, Any]) -> None:
    if not bool(report.get("ok")):
        raise StageDValidationError("\n".join(str(issue) for issue in report.get("issues", [])))


def _load_surrogate_predictions(allocator_run_dir: Path) -> list[dict[str, Any]]:
    predictions_path = allocator_run_dir / "surrogate" / "surrogate_predictions.json"
    payload = _read_json(predictions_path)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise StageDValidationError(f"{predictions_path} must define a non-empty records list")
    return [dict(record) for record in records]


def _load_allocation_manifest(allocator_run_dir: Path) -> dict[str, Any]:
    manifest_path = allocator_run_dir / "allocation" / "allocation_manifest.json"
    return _read_json(manifest_path)


def _record_lookup(records: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        lookup[(str(record["cell_id"]), int(record["candidate_rank"]))] = dict(record)
    return lookup


def _sorted_candidate_ranks(records: Sequence[Mapping[str, Any]]) -> list[int]:
    return sorted({int(record["candidate_rank"]) for record in records})


def _sorted_cell_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(record["cell_id"]) for record in records})


def _max_rank(records: Sequence[Mapping[str, Any]]) -> int:
    candidate_ranks = _sorted_candidate_ranks(records)
    if not candidate_ranks:
        raise StageDValidationError("At least one candidate rank is required")
    return max(candidate_ranks)


def _cell_utility_at_rank(
    records: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    rank: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for record in records:
        if int(record["candidate_rank"]) != rank:
            continue
        result[str(record["cell_id"])] = _round_float(float(record[value_field]))
    return result


def _measured_cell_utility(records: Sequence[Mapping[str, Any]], *, value_field: str) -> dict[str, float]:
    return _cell_utility_at_rank(records, value_field=value_field, rank=_max_rank(records))


def _top_mass_fraction(cell_utilities: Mapping[str, float], fraction: float) -> dict[str, Any]:
    ordered_items = sorted(cell_utilities.items(), key=lambda item: (-float(item[1]), item[0]))
    if not ordered_items:
        raise StageDValidationError("Cell utility mapping must not be empty")
    top_count = max(1, int(math.ceil(len(ordered_items) * float(fraction))))
    total_mass = sum(float(value) for _, value in ordered_items)
    explained_mass = sum(float(value) for _, value in ordered_items[:top_count])
    ratio = 0.0 if total_mass <= 0.0 else _round_float(explained_mass / total_mass)
    return {
        "cell_count": len(ordered_items),
        "top_count": top_count,
        "total_mass": _round_float(total_mass),
        "explained_mass": _round_float(explained_mass),
        "ratio": ratio,
        "top_cells": [cell_id for cell_id, _ in ordered_items[:top_count]],
    }


def _stable_holdout_order(cell_id: str, seed: int) -> tuple[int, str]:
    digest = hashlib.sha256(f"{seed}:{cell_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False), cell_id


def _split_cells(cell_ids: Sequence[str], *, holdout_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    ordered = sorted((str(cell_id) for cell_id in cell_ids), key=lambda item: _stable_holdout_order(item, seed))
    holdout_count = max(1, int(math.ceil(len(ordered) * float(holdout_fraction))))
    holdout_ids = ordered[:holdout_count]
    train_ids = ordered[holdout_count:]
    if not train_ids:
        raise StageDValidationError("Holdout split left no training cells for the surrogate fit")
    return train_ids, holdout_ids


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    pairs = sorted(((float(value), index) for index, value in enumerate(values)), key=lambda item: (item[0], item[1]))
    ranks = np.zeros(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(pairs):
        end = cursor + 1
        while end < len(pairs) and pairs[end][0] == pairs[cursor][0]:
            end += 1
        average_rank = (cursor + end - 1) / 2.0
        for _, index in pairs[cursor:end]:
            ranks[index] = average_rank
        cursor = end
    return ranks


def _spearman_rho(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or not x_values:
        raise StageDValidationError("Spearman inputs must be non-empty and equal in length")
    x_ranks = _average_ranks(x_values)
    y_ranks = _average_ranks(y_values)
    x_centered = x_ranks - np.mean(x_ranks)
    y_centered = y_ranks - np.mean(y_ranks)
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= 0.0:
        return 0.0
    return _round_float(float(np.dot(x_centered, y_centered) / denominator))


def _precision_at_k(predicted: Mapping[str, float], measured: Mapping[str, float], k: int) -> dict[str, Any]:
    top_k = max(1, min(int(k), len(predicted), len(measured)))
    predicted_top = [item[0] for item in sorted(predicted.items(), key=lambda item: (-float(item[1]), item[0]))[:top_k]]
    measured_top = [item[0] for item in sorted(measured.items(), key=lambda item: (-float(item[1]), item[0]))[:top_k]]
    overlap = sorted(set(predicted_top) & set(measured_top))
    return {
        "k": top_k,
        "precision": _round_float(len(overlap) / float(top_k)),
        "predicted_top_cells": predicted_top,
        "measured_top_cells": measured_top,
        "overlap_cells": overlap,
    }


def compute_surrogate_holdout_metrics(
    utility_records: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    holdout_fraction: float,
    top_k_cells: int,
    seed: int,
) -> dict[str, Any]:
    all_cell_ids = _sorted_cell_ids(utility_records)
    train_cells, holdout_cells = _split_cells(all_cell_ids, holdout_fraction=holdout_fraction, seed=seed)
    train_records = [dict(record) for record in utility_records if str(record["cell_id"]) in set(train_cells)]
    holdout_records = [dict(record) for record in utility_records if str(record["cell_id"]) in set(holdout_cells)]
    fit_result = fit_linear_surrogate(train_records)
    scored_holdout = score_records(fit_result["model"], holdout_records)
    max_rank = _max_rank(scored_holdout)
    predicted = _cell_utility_at_rank(scored_holdout, value_field="predicted_utility", rank=max_rank)
    measured = _cell_utility_at_rank(scored_holdout, value_field=value_field, rank=max_rank)
    common_cells = sorted(set(predicted) & set(measured))
    rho = _spearman_rho(
        [predicted[cell_id] for cell_id in common_cells],
        [measured[cell_id] for cell_id in common_cells],
    )
    precision = _precision_at_k(predicted, measured, top_k_cells)
    return {
        "train_cell_count": len(train_cells),
        "holdout_cell_count": len(holdout_cells),
        "holdout_cell_ids": holdout_cells,
        "spearman_rho": rho,
        "top_k_precision": precision["precision"],
        "top_k": precision["k"],
        "top_k_overlap_cells": precision["overlap_cells"],
        "predicted_top_cells": precision["predicted_top_cells"],
        "measured_top_cells": precision["measured_top_cells"],
    }


def _augment_selection(
    record: Mapping[str, Any],
    *,
    measured_lookup: Mapping[tuple[str, int], Mapping[str, Any]],
    predicted_lookup: Mapping[tuple[str, int], Mapping[str, Any]],
    value_field: str,
) -> dict[str, Any]:
    cell_id = str(record["cell_id"])
    candidate_rank = int(record["candidate_rank"])
    measured_record = dict(measured_lookup[(cell_id, candidate_rank)])
    predicted_record = dict(predicted_lookup[(cell_id, candidate_rank)])
    return {
        "cell_id": cell_id,
        "layer_group_id": str(measured_record["layer_group_id"]),
        "timestep_band_id": str(measured_record["timestep_band_id"]),
        "candidate_rank": candidate_rank,
        "cost": int(candidate_rank),
        "predicted_utility": _round_float(float(predicted_record["predicted_utility"])),
        "measured_utility": _round_float(float(measured_record[value_field])),
        "is_attention_cell": bool(measured_record["is_attention_cell"]),
        "rank_fraction_of_max": _round_float(float(measured_record["rank_fraction_of_max"])),
        "layer_count": int(measured_record["layer_count"]),
        "step_count": int(measured_record["step_count"]),
        "event_count": int(measured_record["event_count"]),
    }


def _summarize_method(
    method: str,
    selections: Sequence[Mapping[str, Any]],
    *,
    budget: int,
) -> dict[str, Any]:
    used_budget = sum(int(selection["cost"]) for selection in selections)
    return {
        "method": method,
        "status": "available",
        "selected_cell_count": len(selections),
        "nonzero_cell_count": sum(1 for selection in selections if int(selection["candidate_rank"]) > 0),
        "used_budget": int(used_budget),
        "available_budget": int(budget),
        "budget_utilization": _round_float(used_budget / float(budget)),
        "total_predicted_utility": _round_float(sum(float(selection["predicted_utility"]) for selection in selections)),
        "total_measured_utility": _round_float(sum(float(selection["measured_utility"]) for selection in selections)),
        "selections": [dict(selection) for selection in selections],
    }


def _derive_uniform_method(
    predicted_records: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    measured_lookup: Mapping[tuple[str, int], Mapping[str, Any]],
    predicted_lookup: Mapping[tuple[str, int], Mapping[str, Any]],
    value_field: str,
) -> dict[str, Any]:
    cell_ids = _sorted_cell_ids(predicted_records)
    candidate_ranks = _sorted_candidate_ranks(predicted_records)
    best_rank: int | None = None
    best_predicted_total = -1.0
    for candidate_rank in candidate_ranks:
        cost = len(cell_ids) * int(candidate_rank)
        if cost > budget:
            continue
        predicted_total = sum(float(predicted_lookup[(cell_id, candidate_rank)]["predicted_utility"]) for cell_id in cell_ids)
        if predicted_total > best_predicted_total:
            best_predicted_total = predicted_total
            best_rank = int(candidate_rank)
            continue
        if predicted_total == best_predicted_total and best_rank is not None and int(candidate_rank) < best_rank:
            best_rank = int(candidate_rank)
    if best_rank is None:
        raise StageDValidationError("No feasible uniform rank satisfies the requested budget")
    selections = [
        _augment_selection(
            {"cell_id": cell_id, "candidate_rank": best_rank},
            measured_lookup=measured_lookup,
            predicted_lookup=predicted_lookup,
            value_field=value_field,
        )
        for cell_id in cell_ids
    ]
    return _summarize_method("uniform", selections, budget=budget)


def _solve_shared_rank_method(
    predicted_records: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    group_field: str,
    method: str,
    measured_lookup: Mapping[tuple[str, int], Mapping[str, Any]],
    predicted_lookup: Mapping[tuple[str, int], Mapping[str, Any]],
    value_field: str,
) -> dict[str, Any]:
    group_to_cells: dict[str, list[str]] = {}
    group_to_attention: dict[str, bool] = {}
    option_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    max_rank = _max_rank(predicted_records)
    for record in predicted_records:
        group_id = str(record[group_field])
        cell_id = str(record["cell_id"])
        group_to_cells.setdefault(group_id, [])
        if cell_id not in group_to_cells[group_id]:
            group_to_cells[group_id].append(cell_id)
        group_to_attention[group_id] = group_to_attention.get(group_id, False) or bool(record["is_attention_cell"])
        key = (group_id, int(record["candidate_rank"]))
        option = option_lookup.setdefault(
            key,
            {
                "cell_id": group_id,
                "layer_group_id": group_id if group_field == "layer_group_id" else str(record["layer_group_id"]),
                "timestep_band_id": group_id if group_field == "timestep_band_id" else str(record["timestep_band_id"]),
                "candidate_rank": int(record["candidate_rank"]),
                "rank_fraction_of_max": _round_float(int(record["candidate_rank"]) / float(max_rank)),
                "layer_count": 0,
                "step_count": 0,
                "event_count": 0,
                "cost": 0,
                "utility": 0.0,
            },
        )
        option["utility"] = _round_float(float(option["utility"]) + float(record["predicted_utility"]))
        option["cost"] = int(record["candidate_rank"]) * len(group_to_cells[group_id])
        option["layer_count"] = int(option["layer_count"]) + int(record["layer_count"])
        option["step_count"] = int(option["step_count"]) + int(record["step_count"])
        option["event_count"] = int(option["event_count"]) + int(record["event_count"])

    choice_groups = [
        {
            "group_id": group_id,
            "layer_group_id": group_id if group_field == "layer_group_id" else None,
            "timestep_band_id": group_id if group_field == "timestep_band_id" else None,
            "is_attention_cell": group_to_attention.get(group_id, False),
            "options": sorted(
                [dict(option_lookup[(group_id, candidate_rank)]) for candidate_rank in _sorted_candidate_ranks(predicted_records)],
                key=lambda item: (int(item["cost"]), -float(item["utility"]), int(item["candidate_rank"])),
            ),
        }
        for group_id in sorted(group_to_cells)
    ]
    result = solve_multiple_choice_knapsack(choice_groups, budget=budget)
    selections: list[dict[str, Any]] = []
    for group_selection in result["selections"]:
        group_id = str(group_selection["cell_id"])
        candidate_rank = int(group_selection["candidate_rank"])
        for cell_id in sorted(group_to_cells[group_id]):
            selections.append(
                _augment_selection(
                    {"cell_id": cell_id, "candidate_rank": candidate_rank},
                    measured_lookup=measured_lookup,
                    predicted_lookup=predicted_lookup,
                    value_field=value_field,
                )
            )
    return _summarize_method(method, selections, budget=budget)


def _derive_proposed_method(
    predicted_records: Sequence[Mapping[str, Any]],
    *,
    allocator_run_dir: Path,
    budget: int,
    measured_lookup: Mapping[tuple[str, int], Mapping[str, Any]],
    predicted_lookup: Mapping[tuple[str, int], Mapping[str, Any]],
    value_field: str,
) -> dict[str, Any]:
    manifest = _load_allocation_manifest(allocator_run_dir)
    manifest_budget = None
    if isinstance(manifest.get("budget"), dict) and manifest["budget"].get("available") not in (None, ""):
        manifest_budget = int(manifest["budget"]["available"])
    if manifest_budget == budget and isinstance(manifest.get("selections"), list):
        selections = [
            _augment_selection(
                {"cell_id": str(item["cell_id"]), "candidate_rank": int(item["candidate_rank"])},
                measured_lookup=measured_lookup,
                predicted_lookup=predicted_lookup,
                value_field=value_field,
            )
            for item in manifest["selections"]
        ]
    else:
        choice_groups = build_choice_groups(
            predicted_records,
            cost_field="candidate_rank",
            utility_field="predicted_utility",
        )
        result = solve_multiple_choice_knapsack(choice_groups, budget=budget)
        selections = [
            _augment_selection(
                {
                    "cell_id": str(item["cell_id"]),
                    "candidate_rank": int(item["candidate_rank"]),
                },
                measured_lookup=measured_lookup,
                predicted_lookup=predicted_lookup,
                value_field=value_field,
            )
            for item in result["selections"]
        ]
    return _summarize_method("proposed", selections, budget=budget)


def build_method_allocations(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    probe_run_dir = Path(config["paths"]["probe_run_dir"])
    allocator_run_dir = Path(config["paths"]["allocator_run_dir"])
    utility_records = load_utility_records(probe_run_dir / "utility_records.json")
    predicted_records = _load_surrogate_predictions(allocator_run_dir)
    measured_lookup = _record_lookup(utility_records)
    predicted_lookup = _record_lookup(predicted_records)
    budget = int(config["allocation"]["budget"])
    value_field = str(config["allocation"]["utility_field"])

    methods: dict[str, dict[str, Any]] = {
        "uniform": _derive_uniform_method(
            predicted_records,
            budget=budget,
            measured_lookup=measured_lookup,
            predicted_lookup=predicted_lookup,
            value_field=value_field,
        ),
        "layer_only": _solve_shared_rank_method(
            predicted_records,
            budget=budget,
            group_field="layer_group_id",
            method="layer_only",
            measured_lookup=measured_lookup,
            predicted_lookup=predicted_lookup,
            value_field=value_field,
        ),
        "timestep_only": _solve_shared_rank_method(
            predicted_records,
            budget=budget,
            group_field="timestep_band_id",
            method="timestep_only",
            measured_lookup=measured_lookup,
            predicted_lookup=predicted_lookup,
            value_field=value_field,
        ),
        "proposed": _derive_proposed_method(
            predicted_records,
            allocator_run_dir=allocator_run_dir,
            budget=budget,
            measured_lookup=measured_lookup,
            predicted_lookup=predicted_lookup,
            value_field=value_field,
        ),
        "t_lora": {
            "method": "t_lora",
            "status": "unavailable",
            "reason": "Exact matched-budget T-LoRA projection is not implemented in the repo-local wrapper.",
            "used_budget": None,
            "available_budget": budget,
            "budget_utilization": None,
            "selected_cell_count": None,
            "nonzero_cell_count": None,
            "total_predicted_utility": None,
            "total_measured_utility": None,
            "selections": [],
        },
    }
    return methods


def _gate_threshold_row(name: str, measured: float | None, required: float, *, comparator: str) -> dict[str, Any]:
    passed = None
    if measured is not None:
        if comparator == "ge":
            passed = bool(measured >= required)
        elif comparator == "le":
            passed = bool(measured <= required)
        else:
            raise StageDValidationError(f"Unsupported comparator {comparator!r}")
    return {
        "name": name,
        "measured": measured,
        "required": required,
        "comparator": comparator,
        "passed": passed,
    }


def _load_training_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _read_json(path)


def _training_index(training_summary: Mapping[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    if not training_summary:
        return {}
    task_runs = training_summary.get("task_runs")
    if not isinstance(task_runs, list):
        return {}
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for item in task_runs:
        if not isinstance(item, dict):
            continue
        index[(str(item.get("task_id")), str(item.get("method")))] = dict(item)
    return index


def evaluate_stage_d_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    probe_run_dir = Path(config["paths"]["probe_run_dir"])
    utility_records = load_utility_records(probe_run_dir / "utility_records.json")
    training_summary = _load_training_summary(Path(config["paths"]["training_summary_json"]))
    training_runs = _training_index(training_summary)
    method_allocations = build_method_allocations(config)
    measured_cell_utility = _measured_cell_utility(
        utility_records,
        value_field=str(config["allocation"]["utility_field"]),
    )
    top_mass = _top_mass_fraction(measured_cell_utility, float(config["allocation"]["top_mass_fraction"]))
    holdout = compute_surrogate_holdout_metrics(
        utility_records,
        value_field=str(config["allocation"]["utility_field"]),
        holdout_fraction=float(config["allocation"]["holdout_fraction"]),
        top_k_cells=int(config["allocation"]["top_k_cells"]),
        seed=int(config["run"]["seed"]),
    )

    baseline_rows: list[dict[str, Any]] = []
    uniform_metric = float(method_allocations["uniform"]["total_measured_utility"])
    best_marginal_metric = max(
        float(method_allocations["layer_only"]["total_measured_utility"]),
        float(method_allocations["timestep_only"]["total_measured_utility"]),
    )
    proposed_metric = float(method_allocations["proposed"]["total_measured_utility"])
    for method in METHOD_ORDER:
        payload = dict(method_allocations[method])
        row = {
            "method": method,
            "status": payload.get("status"),
            "used_budget": payload.get("used_budget"),
            "available_budget": payload.get("available_budget"),
            "budget_utilization": payload.get("budget_utilization"),
            "selected_cell_count": payload.get("selected_cell_count"),
            "nonzero_cell_count": payload.get("nonzero_cell_count"),
            "total_predicted_utility": payload.get("total_predicted_utility"),
            "total_measured_utility": payload.get("total_measured_utility"),
            "relative_vs_uniform": None,
            "task_run_statuses": "",
            "notes": payload.get("reason", ""),
        }
        if payload.get("total_measured_utility") is not None and uniform_metric > 0.0:
            row["relative_vs_uniform"] = _round_float(
                (float(payload["total_measured_utility"]) - uniform_metric) / uniform_metric
            )
        task_statuses = []
        for task_id in config["pilot"]["task_ids"]:
            task_run = training_runs.get((task_id, method))
            if task_run:
                task_statuses.append(f"{task_id}:{task_run.get('status')}")
        row["task_run_statuses"] = ";".join(task_statuses)
        baseline_rows.append(row)

    overhead_ratio = None
    if training_summary:
        overhead_ratio = training_summary.get("probe_allocation_overhead_ratio")
        if overhead_ratio not in (None, ""):
            overhead_ratio = _round_float(float(overhead_ratio))
        else:
            overhead_ratio = None

    thresholds = dict(config["gate"]["thresholds"])
    threshold_rows = [
        _gate_threshold_row("top_20_mass_ratio", float(top_mass["ratio"]), float(thresholds["top_20_mass_min"]), comparator="ge"),
        _gate_threshold_row("held_out_spearman_rho", float(holdout["spearman_rho"]), float(thresholds["held_out_spearman_min"]), comparator="ge"),
        _gate_threshold_row("top_5_precision", float(holdout["top_k_precision"]), float(thresholds["top_5_precision_min"]), comparator="ge"),
        _gate_threshold_row(
            "proposed_relative_vs_uniform",
            _round_float((proposed_metric - uniform_metric) / uniform_metric) if uniform_metric > 0.0 else None,
            float(thresholds["proposed_relative_gain_min"]),
            comparator="ge",
        ),
        _gate_threshold_row(
            "proposed_vs_best_marginal",
            _round_float(proposed_metric - best_marginal_metric),
            0.0,
            comparator="ge",
        ),
        _gate_threshold_row(
            "probe_allocation_overhead_ratio",
            overhead_ratio,
            float(thresholds["overhead_ratio_max"]),
            comparator="le",
        ),
    ]

    blockers: list[str] = []
    if not training_summary:
        blockers.append("No Stage D training summary was found, so GPU-backed pilot training was not verified.")
    else:
        blocked_runs = [
            f"{task_id}:{method}:{run.get('status')}"
            for (task_id, method), run in sorted(training_runs.items())
            if str(run.get("status")) not in {"planned", "completed", "skipped"}
        ]
        blockers.extend(blocked_runs)

    for method in ("layer_only", "timestep_only", "proposed", "t_lora"):
        if method_allocations[method]["status"] != "available":
            blockers.append(f"{method}: {method_allocations[method].get('reason')}")

    threshold_pass_values = [row["passed"] for row in threshold_rows]
    decision = "GO" if threshold_pass_values and all(value is True for value in threshold_pass_values) else "NO_GO"

    payload = {
        "schema_version": 1,
        "decision": decision,
        "decision_reason": (
            "All week-2 gate thresholds passed."
            if decision == "GO"
            else "At least one gate threshold failed or remained unmeasured."
        ),
        "locked_metric_name": str(config["evaluation"]["locked_metric_name"]),
        "locked_metric_kind": str(config["evaluation"]["locked_metric_kind"]),
        "source_probe_run_dir": display_path(Path(config["paths"]["probe_run_dir"]), Path(config["paths"]["repo_root"])),
        "source_allocator_run_dir": display_path(
            Path(config["paths"]["allocator_run_dir"]),
            Path(config["paths"]["repo_root"]),
        ),
        "preflight": collect_preflight_report(config),
        "measured": {
            "top_20_mass_ratio": float(top_mass["ratio"]),
            "held_out_spearman_rho": float(holdout["spearman_rho"]),
            "top_5_precision": float(holdout["top_k_precision"]),
            "proposed_relative_vs_uniform": (
                _round_float((proposed_metric - uniform_metric) / uniform_metric) if uniform_metric > 0.0 else None
            ),
            "proposed_minus_best_marginal": _round_float(proposed_metric - best_marginal_metric),
            "probe_allocation_overhead_ratio": overhead_ratio,
        },
        "thresholds": threshold_rows,
        "top_mass": top_mass,
        "holdout": holdout,
        "baseline_table": baseline_rows,
        "methods": method_allocations,
        "training_summary": training_summary,
        "artifact_inventory": sorted(
            {
                display_path(Path(config["paths"]["probe_run_dir"]), Path(config["paths"]["repo_root"])),
                display_path(Path(config["paths"]["allocator_run_dir"]), Path(config["paths"]["repo_root"])),
                display_path(Path(config["paths"]["evaluation_summary_json"]), Path(config["paths"]["repo_root"])),
                display_path(Path(config["paths"]["baseline_table_csv"]), Path(config["paths"]["repo_root"])),
                display_path(Path(config["paths"]["training_summary_json"]), Path(config["paths"]["repo_root"])),
            }
        ),
        "blockers": blockers,
    }
    return payload


def write_stage_d_evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    payload = evaluate_stage_d_gate(config)
    evaluation_summary_path = Path(config["paths"]["evaluation_summary_json"])
    baseline_table_path = Path(config["paths"]["baseline_table_csv"])
    save_json(evaluation_summary_path, payload)
    _write_csv(
        baseline_table_path,
        payload["baseline_table"],
        fieldnames=(
            "method",
            "status",
            "used_budget",
            "available_budget",
            "budget_utilization",
            "selected_cell_count",
            "nonzero_cell_count",
            "total_predicted_utility",
            "total_measured_utility",
            "relative_vs_uniform",
            "task_run_statuses",
            "notes",
        ),
    )
    return {
        "payload": payload,
        "evaluation_summary_path": str(evaluation_summary_path),
        "baseline_table_path": str(baseline_table_path),
    }


def render_gate_memo_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# RD-LoRA week-2 gate memo",
        "",
        f"Decision: **{payload['decision']}**",
        "",
        f"Decision reason: {payload['decision_reason']}",
        "",
        "## Thresholds",
        "",
        "| threshold | measured | requirement | pass |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in payload.get("thresholds", []):
        measured = "n/a" if row["measured"] is None else f"{float(row['measured']):.6f}"
        comparator = ">=" if row["comparator"] == "ge" else "<="
        status = "yes" if row["passed"] is True else "no" if row["passed"] is False else "n/a"
        lines.append(
            f"| {row['name']} | {measured} | {comparator} {float(row['required']):.6f} | {status} |"
        )

    lines.extend(
        [
            "",
            "## Baselines",
            "",
            "| method | status | used budget | measured utility | rel vs uniform | notes |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload.get("baseline_table", []):
        used_budget = "n/a" if row["used_budget"] is None else str(row["used_budget"])
        measured_utility = "n/a" if row["total_measured_utility"] is None else f"{float(row['total_measured_utility']):.6f}"
        relative = "n/a" if row["relative_vs_uniform"] is None else f"{float(row['relative_vs_uniform']):.6f}"
        lines.append(
            f"| {row['method']} | {row['status']} | {used_budget} | {measured_utility} | {relative} | {row['notes']} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
        ]
    )
    for item in payload.get("artifact_inventory", []):
        lines.append(f"- `{item}`")

    blockers = [str(item) for item in payload.get("blockers", []) if str(item).strip()]
    if blockers:
        lines.extend(
            [
                "",
                "## Blockers",
                "",
            ]
        )
        for item in blockers:
            lines.append(f"- {item}")

    return "\n".join(lines) + "\n"


def write_gate_memo(
    payload: Mapping[str, Any],
    *,
    memo_md_path: Path,
    memo_json_path: Path,
) -> dict[str, Any]:
    markdown = render_gate_memo_markdown(payload)
    memo_md_path.parent.mkdir(parents=True, exist_ok=True)
    memo_md_path.write_text(markdown, encoding="utf-8")
    save_json(memo_json_path, dict(payload))
    return {
        "memo_md_path": str(memo_md_path),
        "memo_json_path": str(memo_json_path),
        "payload": dict(payload),
    }


__all__ = [
    "BASELINE_TABLE_FILENAME",
    "DEFAULT_STAGE_D_CONFIG_PATH",
    "EVAL_SUMMARY_FILENAME",
    "METHOD_ORDER",
    "TRAINING_SUMMARY_FILENAME",
    "StageDValidationError",
    "build_method_allocations",
    "collect_preflight_report",
    "default_stage_d_config",
    "ensure_preflight_ok",
    "evaluate_stage_d_gate",
    "render_gate_memo_markdown",
    "resolve_stage_d_config",
    "write_gate_memo",
    "write_stage_d_evaluation",
]
