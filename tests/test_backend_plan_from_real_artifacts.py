from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.runtime.allocation_manifest import load_allocation_manifest, validate_allocation_manifest
from rd_lora.training.adapter_factory import ALLOWED_TARGET_MODULES, build_backend_plan
from rd_lora.training.timestep_routing import resolve_active_adapter_name


MANIFEST_DIR = REPO_ROOT / "outputs/rd_lora/allocation/gate_real"


@pytest.mark.parametrize("backend_name", ["uniform", "layer_only", "timestep_only", "proposed"])
def test_build_backend_plan_from_real_manifest(backend_name: str) -> None:
    manifest = load_allocation_manifest(MANIFEST_DIR / f"{backend_name}.json")
    validate_allocation_manifest(manifest)

    plan = build_backend_plan(
        manifest=manifest,
        task="subject_personalization",
        config={},
    )

    assert plan["backend"] == backend_name
    assert plan["use_rslora"] is True
    assert tuple(plan["target_modules"]) == ALLOWED_TARGET_MODULES
    assert all(isinstance(cell_id, str) for cell_id in plan["cell_ids"])
    assert len(plan["cell_ids"]) == 24

    if backend_name == "uniform":
        assert len(plan["adapter_banks"]) == 1
        bank = plan["adapter_banks"][0]
        assert bank["rank"] == 4
        assert bank["alpha"] == 4
        assert plan["routing"] == {
            "mode": "static",
            "default_adapter_name": "uniform_bank",
        }

    elif backend_name == "layer_only":
        assert len(plan["adapter_banks"]) == 1
        bank = plan["adapter_banks"][0]
        assert "rank" not in bank
        assert len(set(bank["rank_pattern"].values())) > 1
        assert plan["routing"] == {
            "mode": "static",
            "default_adapter_name": "layer_only_bank",
        }

    elif backend_name == "timestep_only":
        assert len(plan["adapter_banks"]) == 4
        assert plan["routing"]["mode"] == "timestep_band"
        for bank in plan["adapter_banks"]:
            assert len(set(bank["rank_pattern"].values())) == 1
            assert len(set(bank["alpha_pattern"].values())) == 1
            assert resolve_active_adapter_name(
                plan["routing"]["table"],
                step_index=plan["routing"]["table"][bank["timestep_band"]]["step_indices"][0],
            ) == bank["adapter_name"]

    else:
        assert len(plan["adapter_banks"]) == 4
        assert plan["routing"]["mode"] == "timestep_band"
        assert any(len(set(bank["rank_pattern"].values())) > 1 for bank in plan["adapter_banks"])
        for bank in plan["adapter_banks"]:
            assert resolve_active_adapter_name(
                plan["routing"]["table"],
                step_index=plan["routing"]["table"][bank["timestep_band"]]["step_indices"][0],
            ) == bank["adapter_name"]
