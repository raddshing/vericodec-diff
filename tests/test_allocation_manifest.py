from __future__ import annotations

from pathlib import Path

import pytest


from rd_lora.runtime.allocation_manifest import (
    AllocationManifestError,
    load_allocation_manifest,
    validate_allocation_manifest,
    write_allocation_manifest,
)


def _manifest() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "backend": "uniform",
        "rank_budget_total": 16,
        "layer_groups": ["layer_group_00"],
        "timestep_bands": ["timestep_band_00"],
        "cells": [
            {
                "cell_id": "layer_group_00__timestep_band_00",
                "layer_group": "layer_group_00",
                "timestep_band": "timestep_band_00",
                "rank": 4,
                "alpha": 4,
                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
                "adapter_name": "uniform_bank",
            }
        ],
    }


def test_write_and_load_allocation_manifest_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "uniform.json"
    write_allocation_manifest(_manifest(), path)
    payload = load_allocation_manifest(path)
    assert payload["schema_version"] == "1.0"
    assert payload["backend"] == "uniform"
    assert payload["cells"][0]["rank"] == 4


def test_validate_allocation_manifest_rejects_budget_overflow() -> None:
    payload = _manifest()
    payload["rank_budget_total"] = 2
    with pytest.raises(AllocationManifestError):
        validate_allocation_manifest(payload)
