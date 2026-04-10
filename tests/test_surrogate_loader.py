from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.surrogate import (
    REQUIRED_PROBE_COLUMNS,
    SURROGATE_FEATURES_V1,
    build_feature_matrix,
    load_probe_dataframe,
)


REAL_PROBE_CSV = REPO_ROOT / "outputs" / "rd_lora" / "probe" / "task1_real" / "cell_utility.csv"


def test_real_probe_loader_matches_schema_and_feature_shape() -> None:
    frame = load_probe_dataframe(REAL_PROBE_CSV)

    assert REAL_PROBE_CSV.is_file()
    assert set(REQUIRED_PROBE_COLUMNS).issubset(frame.columns)
    assert {"rank_fraction_of_max", "layer_group_id", "timestep_band_id", "task_id"}.issubset(frame.columns)

    max_candidate_rank = max(int(frame["candidate_rank"].max()), 1)
    expected_rank_fraction = frame["candidate_rank"].to_numpy(dtype=np.float64) / float(max_candidate_rank)
    assert np.allclose(frame["rank_fraction_of_max"].to_numpy(dtype=np.float64), expected_rank_fraction)

    feature_matrix = build_feature_matrix(frame, SURROGATE_FEATURES_V1)
    assert feature_matrix.shape == (len(frame), len(SURROGATE_FEATURES_V1))
