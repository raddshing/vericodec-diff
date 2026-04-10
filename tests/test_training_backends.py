from __future__ import annotations

from rd_lora.cells import build_cell_schema
from rd_lora.training.adapter_factory import build_backend_adapter_plan


def _manifest_for_backend(backend: str) -> dict[str, object]:
    schema = build_cell_schema()
    cells = []
    for cell in schema.cells:
        if backend == "uniform":
            rank = 4
        elif backend == "layer_only":
            rank = 2 + (cell.layer_group.group_index % 4) * 2
        elif backend == "timestep_only":
            rank = 2 + (cell.timestep_band.band_index * 2)
        else:
            rank = 2 + cell.layer_group.group_index + cell.timestep_band.band_index
        cells.append(
            {
                "cell_id": cell.cell_id,
                "layer_group": cell.layer_group.group_id,
                "timestep_band": cell.timestep_band.band_id,
                "rank": rank,
                "alpha": rank,
                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
                "adapter_name": f"{backend}_{cell.timestep_band.band_id}",
            }
        )
    return {
        "schema_version": "1.0",
        "backend": backend,
        "rank_budget_total": sum(int(cell["rank"]) for cell in cells),
        "layer_groups": [group.group_id for group in schema.layer_groups],
        "timestep_bands": [band.band_id for band in schema.timestep_bands],
        "cells": cells,
    }


def test_uniform_backend_plan_uses_one_adapter_bank() -> None:
    plan = build_backend_adapter_plan("uniform", _manifest_for_backend("uniform"))
    assert plan["use_rslora"] is True
    assert plan["target_modules"] == ["to_k", "to_q", "to_v", "to_out.0"]
    assert len(plan["adapter_banks"]) == 1
    assert set(plan["adapter_banks"][0]["rank_pattern"].values()) == {4}


def test_layer_only_backend_plan_has_heterogeneous_rank_pattern() -> None:
    plan = build_backend_adapter_plan("layer_only", _manifest_for_backend("layer_only"))
    assert len(plan["adapter_banks"]) == 1
    assert len(set(plan["adapter_banks"][0]["rank_pattern"].values())) > 1


def test_timestep_only_backend_plan_uses_one_bank_per_band() -> None:
    plan = build_backend_adapter_plan("timestep_only", _manifest_for_backend("timestep_only"))
    assert len(plan["adapter_banks"]) == 4
    for bank in plan["adapter_banks"]:
        assert len(set(bank["rank_pattern"].values())) == 1
    assert len(plan["routing"]["adapter_step_map"]) == 20


def test_proposed_backend_plan_varies_rank_pattern_within_band() -> None:
    plan = build_backend_adapter_plan("proposed", _manifest_for_backend("proposed"))
    assert len(plan["adapter_banks"]) == 4
    assert any(len(set(bank["rank_pattern"].values())) > 1 for bank in plan["adapter_banks"])
