from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.training.adapter_factory import (
    EXPECTED_BACKENDS,
    REQUIRED_PROBE_ROW_KEYS,
    REQUIRED_PROBE_SUMMARY_KEYS,
    REQUIRED_SURROGATE_SUMMARY_KEYS,
    load_actual_contract_binding,
)


def test_actual_contract_binding_matches_real_artifacts() -> None:
    contract = load_actual_contract_binding()

    assert set(contract) == {"probe", "surrogate", "allocation"}
    assert tuple(contract["allocation"]["backends"]) == EXPECTED_BACKENDS
    assert contract["surrogate"]["model_size_bytes"] > 0

    for task_name in ("subject_personalization", "style_domain"):
        probe = contract["probe"][task_name]
        assert probe["row_source"] == "rows"
        assert set(REQUIRED_PROBE_ROW_KEYS) <= set(probe["row_keys"])
        assert set(REQUIRED_PROBE_SUMMARY_KEYS) <= set(probe["summary_keys"])
        assert probe["row_count"] > 0

    assert set(REQUIRED_SURROGATE_SUMMARY_KEYS) <= set(contract["surrogate"]["summary_keys"])


def test_actual_allocation_manifests_validate_with_expected_schema() -> None:
    contract = load_actual_contract_binding()
    manifests = contract["allocation"]["manifests"]

    for backend_name in EXPECTED_BACKENDS:
        manifest = manifests[backend_name]
        assert manifest["layer_groups"] == [
            "layer_group_00",
            "layer_group_01",
            "layer_group_02",
            "layer_group_03",
            "layer_group_04",
            "layer_group_05",
        ]
        assert manifest["timestep_bands"] == [
            "timestep_band_00",
            "timestep_band_01",
            "timestep_band_02",
            "timestep_band_03",
        ]
        assert manifest["cell_count"] == 24
