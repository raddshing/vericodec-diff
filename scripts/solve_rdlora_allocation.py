from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.allocator import solve_multiple_choice_knapsack
from rd_lora.cells import build_cell_schema
from rd_lora.runtime.allocation_manifest import write_allocation_manifest
from rd_lora.substrate.diffusers_sdxl import deep_update, load_yaml_mapping, save_json
from rd_lora.surrogate import score_records
from vericodec_diff.config import OmegaConf


DEFAULT_CONFIG_PATH = "configs/rdlora_vanilla.yaml"
TARGET_MODULES = ["to_k", "to_q", "to_v", "to_out.0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the RD-LoRA rank allocation and emit standardized manifest artifacts for all backends."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config path.")
    parser.add_argument("--probe_dir", required=True, help="Probe output directory with cell_utility.json.")
    parser.add_argument("--surrogate_dir", required=True, help="Surrogate output directory with surrogate_model.pkl.")
    parser.add_argument("--output_dir", required=True, help="Output directory for allocation artifacts.")
    return parser.parse_args()


def _default_config() -> dict[str, Any]:
    return {
        "allocation": {
            "rank_budget_total": 96,
            "utility_scale": 1000000,
            "max_exact_state_count": 500000,
        }
    }


def _load_config(path_value: str) -> dict[str, Any]:
    config_path = Path(path_value).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    raw = load_yaml_mapping(config_path) if config_path.is_file() else {}
    return deep_update(_default_config(), raw)


def _load_probe_rows(probe_dir: Path) -> list[dict[str, Any]]:
    payload = load_yaml_mapping(probe_dir / "cell_utility.json")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{probe_dir / 'cell_utility.json'} must contain a non-empty rows list")
    return [dict(row) for row in rows]


def _load_model(surrogate_dir: Path) -> Any:
    with (surrogate_dir / "surrogate_model.pkl").open("rb") as handle:
        return pickle.load(handle)


def _group_rows(rows: Sequence[Mapping[str, Any]], field_name: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[field_name]), []).append(dict(row))
    return grouped


def _select_uniform_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_ranks: Sequence[int],
    cell_count: int,
    budget_total: int,
) -> dict[str, int]:
    row_by_rank = _group_rows(rows, "candidate_rank")
    best_rank = 0
    best_utility = None
    for rank in candidate_ranks:
        rank_int = int(rank)
        used_budget = rank_int * cell_count
        if used_budget > budget_total:
            continue
        utility = round(sum(float(item["predicted_utility"]) for item in row_by_rank[str(rank_int)]), 6)
        if best_utility is None or utility > best_utility or (utility == best_utility and rank_int < best_rank):
            best_rank = rank_int
            best_utility = utility
    return {cell.cell_id: best_rank for cell in build_cell_schema(candidate_ranks=candidate_ranks).cells}


def _solve_grouped_allocation(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_field: str,
    candidate_ranks: Sequence[int],
    budget_total: int,
    unit_multiplier: int,
) -> dict[str, int]:
    grouped_rows = _group_rows(rows, group_field)
    choice_groups: list[dict[str, Any]] = []
    for group_id, group_rows in sorted(grouped_rows.items()):
        rows_by_rank = _group_rows(group_rows, "candidate_rank")
        options: list[dict[str, Any]] = []
        for rank in candidate_ranks:
            rank_int = int(rank)
            options.append(
                {
                    "cell_id": group_id,
                    "layer_group_id": group_id if group_field == "layer_group_id" else "shared",
                    "timestep_band_id": group_id if group_field == "timestep_band_id" else "shared",
                    "candidate_rank": rank_int,
                    "cost": rank_int * unit_multiplier,
                    "utility": round(
                        sum(float(item["predicted_utility"]) for item in rows_by_rank[str(rank_int)]),
                        6,
                    ),
                    "rank_fraction_of_max": max(0.0, float(rank_int) / float(max(candidate_ranks))),
                    "layer_count": 1,
                    "step_count": 1,
                    "event_count": len(rows_by_rank[str(rank_int)]),
                }
            )
        choice_groups.append({"group_id": group_id, "is_attention_cell": True, "options": options})

    result = solve_multiple_choice_knapsack(
        choice_groups,
        budget=budget_total,
        utility_scale=1000000,
        max_exact_state_count=500000,
    )
    return {str(item["cell_id"]): int(item["candidate_rank"]) for item in result["selections"]}


def _solve_proposed(rows: Sequence[Mapping[str, Any]], *, budget_total: int) -> dict[str, int]:
    choice_groups: list[dict[str, Any]] = []
    by_cell = _group_rows(rows, "cell_id")
    for cell_id, cell_rows in sorted(by_cell.items()):
        options: list[dict[str, Any]] = []
        for row in sorted(cell_rows, key=lambda item: (int(item["candidate_rank"]), str(item["cell_id"]))):
            options.append(
                {
                    "cell_id": str(row["cell_id"]),
                    "layer_group_id": str(row["layer_group_id"]),
                    "timestep_band_id": str(row["timestep_band_id"]),
                    "candidate_rank": int(row["candidate_rank"]),
                    "cost": int(row["candidate_rank"]),
                    "utility": float(row["predicted_utility"]),
                    "rank_fraction_of_max": float(row["rank_fraction_of_max"]),
                    "layer_count": int(row["layer_count"]),
                    "step_count": int(row["step_count"]),
                    "event_count": int(row["event_count"]),
                }
            )
        choice_groups.append({"group_id": cell_id, "is_attention_cell": True, "options": options})
    result = solve_multiple_choice_knapsack(
        choice_groups,
        budget=budget_total,
        utility_scale=1000000,
        max_exact_state_count=500000,
    )
    return {str(item["cell_id"]): int(item["candidate_rank"]) for item in result["selections"]}


def _build_manifest(backend: str, *, rank_budget_total: int, cell_ranks: Mapping[str, int]) -> dict[str, Any]:
    schema = build_cell_schema()
    cells: list[dict[str, Any]] = []
    for cell in schema.cells:
        rank = int(cell_ranks[cell.cell_id])
        if backend in {"uniform", "layer_only"}:
            adapter_name = f"{backend}_bank"
        else:
            adapter_name = f"{backend}__{cell.timestep_band.band_id}"
        cells.append(
            {
                "cell_id": cell.cell_id,
                "layer_group": cell.layer_group.group_id,
                "timestep_band": cell.timestep_band.band_id,
                "rank": rank,
                "alpha": rank,
                "target_modules": list(TARGET_MODULES),
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


def main() -> int:
    args = parse_args()
    started_at = time.monotonic()
    config = _load_config(args.config)
    probe_dir = Path(args.probe_dir).expanduser().resolve()
    surrogate_dir = Path(args.surrogate_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = deep_update(
        config,
        {
            "paths": {
                "repo_root": str(REPO_ROOT),
                "probe_dir": str(probe_dir),
                "surrogate_dir": str(surrogate_dir),
                "output_dir": str(output_dir),
            }
        },
    )
    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(resolved_config), resolved_config_path)

    utility_rows = _load_probe_rows(probe_dir)
    model = _load_model(surrogate_dir)
    scored_rows = score_records(model, utility_rows)
    schema = build_cell_schema()
    candidate_ranks = list(schema.candidate_ranks)
    rank_budget_total = int(resolved_config["allocation"]["rank_budget_total"])

    uniform_ranks = _select_uniform_rows(
        scored_rows,
        candidate_ranks=candidate_ranks,
        cell_count=len(schema.cells),
        budget_total=rank_budget_total,
    )
    layer_group_ranks = _solve_grouped_allocation(
        scored_rows,
        group_field="layer_group_id",
        candidate_ranks=candidate_ranks,
        budget_total=rank_budget_total,
        unit_multiplier=len(schema.timestep_bands),
    )
    layer_only_ranks = {cell.cell_id: int(layer_group_ranks[cell.layer_group.group_id]) for cell in schema.cells}
    timestep_band_ranks = _solve_grouped_allocation(
        scored_rows,
        group_field="timestep_band_id",
        candidate_ranks=candidate_ranks,
        budget_total=rank_budget_total,
        unit_multiplier=len(schema.layer_groups),
    )
    timestep_only_ranks = {cell.cell_id: int(timestep_band_ranks[cell.timestep_band.band_id]) for cell in schema.cells}
    proposed_ranks = _solve_proposed(scored_rows, budget_total=rank_budget_total)

    manifests = {
        "uniform": _build_manifest("uniform", rank_budget_total=rank_budget_total, cell_ranks=uniform_ranks),
        "layer_only": _build_manifest("layer_only", rank_budget_total=rank_budget_total, cell_ranks=layer_only_ranks),
        "timestep_only": _build_manifest(
            "timestep_only",
            rank_budget_total=rank_budget_total,
            cell_ranks=timestep_only_ranks,
        ),
        "proposed": _build_manifest("proposed", rank_budget_total=rank_budget_total, cell_ranks=proposed_ranks),
    }

    manifest_files: dict[str, str] = {}
    for backend, manifest in manifests.items():
        path = output_dir / f"{backend}.json"
        write_allocation_manifest(manifest, path)
        manifest_files[backend] = str(path)

    allocation_summary_path = output_dir / "allocation_summary.json"
    save_json(
        allocation_summary_path,
        {
            "schema_version": "1.0",
            "rank_budget_total": rank_budget_total,
            "manifest_files": manifest_files,
            "backends": list(manifest_files),
            "source_probe_dir": str(probe_dir),
            "source_surrogate_dir": str(surrogate_dir),
            "wall_time_seconds": round(time.monotonic() - started_at, 6),
        },
    )

    print(f"resolved_config={resolved_config_path}")
    print(f"allocation_summary={allocation_summary_path}")
    for backend, path in manifest_files.items():
        print(f"{backend}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
