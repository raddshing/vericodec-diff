from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.cells import DEFAULT_CANDIDATE_RANKS, build_cell_schema, parse_candidate_ranks_argument
from rd_lora.features import (
    REQUIRED_PROBE_ROW_FIELDNAMES,
    REQUIRED_PROBE_SUMMARY_KEYS,
    REQUIRED_PROVENANCE_KEYS,
    UTILITY_RECORD_FIELDNAMES,
)


RUN_SCRIPT = REPO_ROOT / "scripts" / "run_rdlora_probe.py"


def test_cell_schema_is_deterministic_and_has_24_cells() -> None:
    schema_a = build_cell_schema()
    schema_b = build_cell_schema()

    assert schema_a.to_dict() == schema_b.to_dict()
    assert len(schema_a.layer_groups) == 6
    assert len(schema_a.timestep_bands) == 4
    assert len(schema_a.cells) == 24
    assert schema_a.candidate_ranks == DEFAULT_CANDIDATE_RANKS


def test_candidate_rank_parsing_is_deterministic() -> None:
    assert parse_candidate_ranks_argument("16, 4, 0, 8, 2, 8") == DEFAULT_CANDIDATE_RANKS
    assert parse_candidate_ranks_argument((0, 16, 2, 4, 8, 2)) == DEFAULT_CANDIDATE_RANKS


def test_probe_output_schema_matches_required_columns_and_keys(tmp_path: Path) -> None:
    output_dir = tmp_path / "probe"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--config",
            "configs/rdlora_probe.yaml",
            "--task",
            "subject_personalization",
            "--run_mode",
            "cpu_mock",
            "--output_dir",
            str(output_dir),
            "--max_cells",
            "2",
            "--candidate_ranks",
            "0,2",
            "--probe_inner_steps",
            "1",
            "--max_train_batches",
            "1",
            "--max_val_batches",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "resolved_config.yaml").is_file()

    payload = json.loads((output_dir / "cell_utility.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "probe_summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((output_dir / "run_provenance.json").read_text(encoding="utf-8"))

    assert payload["row_count"] == 4
    assert summary["probe_mode"] == "cpu_mock"
    assert summary["row_count"] == 4
    assert summary["cell_count"] == 2
    assert set(REQUIRED_PROBE_SUMMARY_KEYS).issubset(summary)
    assert set(REQUIRED_PROVENANCE_KEYS).issubset(provenance)

    first_row = payload["rows"][0]
    assert set(REQUIRED_PROBE_ROW_FIELDNAMES).issubset(first_row)
    assert first_row["layer_group"] == first_row["layer_group_id"]
    assert first_row["timestep_band"] == first_row["timestep_band_id"]
    assert first_row["utility"] == first_row["utility_score"]

    with (output_dir / "cell_utility.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == UTILITY_RECORD_FIELDNAMES
        rows = list(reader)
    assert len(rows) == 4
