from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.cells import (
    DEFAULT_CANDIDATE_RANKS,
    build_cell_schema,
    default_layer_catalog,
)
from rd_lora.features import UTILITY_RECORD_FIELDNAMES
from rd_lora.probe import validate_probe_outputs


RUN_SCRIPT = REPO_ROOT / "scripts" / "run_rdlora_probe.py"


class RdLoraProbeTests(unittest.TestCase):
    def test_cell_mapping_is_deterministic(self) -> None:
        schema_a = build_cell_schema()
        schema_b = build_cell_schema()

        self.assertEqual(schema_a.to_dict(), schema_b.to_dict())
        first_cell = schema_a.locate_cell(layer_id="down_blocks.0.attentions.0", step_index=0)
        self.assertEqual(first_cell.cell_id, "layer_group_00__timestep_band_00")
        last_cell = schema_a.locate_cell(layer_id="up_blocks.2.attentions.1", step_index=19)
        self.assertEqual(last_cell.cell_id, "layer_group_05__timestep_band_03")

    def test_layer_by_timestep_schema_uses_six_groups_and_four_bands(self) -> None:
        schema = build_cell_schema()

        self.assertEqual(len(schema.layer_groups), 6)
        self.assertEqual(len(schema.timestep_bands), 4)
        self.assertEqual(len(schema.cells), 24)
        self.assertEqual(len(default_layer_catalog()), 16)
        self.assertEqual(schema.timestep_bands[0].step_indices, (0, 1, 2, 3, 4))
        self.assertEqual(schema.timestep_bands[-1].step_indices, (15, 16, 17, 18, 19))

    def test_candidate_rank_schema_matches_week2_requirements(self) -> None:
        schema = build_cell_schema()
        self.assertEqual(schema.candidate_ranks, DEFAULT_CANDIDATE_RANKS)
        for cell in schema.cells:
            self.assertEqual(cell.candidate_ranks, DEFAULT_CANDIDATE_RANKS)

    def test_cpu_mock_smoke_run_writes_valid_json_and_csv_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmpdir:
            output_root = Path(tmpdir) / "probe_outputs"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RUN_SCRIPT),
                    "--config",
                    "configs/rdlora_probe.yaml",
                    "--run-name",
                    "cpu_smoke",
                    "--set",
                    f"paths.output_root={json.dumps(str(output_root))}",
                    "--set",
                    "preflight.require_torch_cuda=false",
                    "--set",
                    "preflight.required_diffusers_version=null",
                    "--set",
                    "backend.mock.sample_count=1",
                    "--set",
                    "backend.mock.vector_size=4",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            run_dir = output_root / "cpu_smoke"
            validation = validate_probe_outputs(run_dir)
            self.assertTrue(validation["ok"])

            summary = json.loads((run_dir / "probe_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["cell_count"], 24)
            self.assertEqual(summary["candidate_rank_count"], 5)
            self.assertEqual(summary["record_count"], 120)
            self.assertTrue(summary["completed_successfully"])
            self.assertIn("schema_validation_ok=1", completed.stdout)

            utility_payload = json.loads((run_dir / "utility_records.json").read_text(encoding="utf-8"))
            self.assertEqual(utility_payload["record_count"], 120)
            first_record = utility_payload["records"][0]
            self.assertIn("attention_output_mean_abs_drift", first_record)
            self.assertIn("utility_score", first_record)

            with (run_dir / "utility_records.csv").open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), UTILITY_RECORD_FIELDNAMES)
                rows = list(reader)
            self.assertEqual(len(rows), 120)


if __name__ == "__main__":
    unittest.main()
