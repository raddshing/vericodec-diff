from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vericodec_diff.patch_metrics import SampleRecord, build_patch_error_payload, write_patch_error_npz
from vericodec_diff.patch_error_targets import augment_patch_error_payload
from vericodec_diff.sparsity_stats import (
    concentration_at_percent,
    connected_component_summary,
    gini_coefficient,
)


SCRIPT_PATH = REPO_ROOT / "scripts" / "compute_sparsity_stats.py"


class SparsityStatsTests(unittest.TestCase):
    def test_single_hot_patch_case_has_expected_concentration_gini_and_components(self) -> None:
        values = np.zeros(256, dtype=np.float32)
        values[0] = 1.0

        self.assertEqual(concentration_at_percent(values, 5), 1.0)
        self.assertEqual(concentration_at_percent(values, 10), 1.0)
        self.assertAlmostEqual(gini_coefficient(values), 255.0 / 256.0)

        components = connected_component_summary(values, 5)
        self.assertEqual(components["active_patch_count"], 1)
        self.assertEqual(components["component_count"], 1)
        self.assertEqual(components["largest_component_size"], 1)
        self.assertEqual(components["mean_component_size"], 1.0)

    def test_compute_sparsity_stats_script_writes_json_csv_and_resolved_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            error_map_root = repo_root / "outputs" / "error_maps" / "kill"
            error_map_root.mkdir(parents=True, exist_ok=True)

            hot = np.zeros(256, dtype=np.float32)
            hot[0] = 1.0
            record = SampleRecord(
                sample_id="sample",
                split="kill",
                source_path=repo_root / "data" / "source" / "sample.png",
                reconstruction_path=repo_root / "data" / "recon" / "sample.png",
                source_local_path="data/source/sample.png",
                reconstruction_local_path="data/recon/sample.png",
                metadata={},
            )
            payload = build_patch_error_payload(
                record,
                {
                    "lpips": hot,
                    "one_minus_ssim": hot * 0.5,
                    "hf_wavelet_l1": hot * 2.0,
                },
                lpips_backend="pixel_l2_debug",
                wavelet="haar",
            )
            payload = augment_patch_error_payload(payload)
            write_patch_error_npz(error_map_root / "sample__patch64.npz", payload)

            command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--repo-root",
                tmpdir,
                "--error-map-root",
                "outputs/error_maps",
                "--phase",
                "smoke_metrics",
                "--metric-names",
                "patch_error",
            ]
            subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            output_dir = repo_root / "outputs" / "metrics" / "smoke_metrics"
            resolved_config_path = output_dir / "resolved_config.yaml"
            summary_path = output_dir / "sparsity_stats.json"
            records_csv_path = output_dir / "sparsity_stats.csv"

            self.assertTrue(resolved_config_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertTrue(records_csv_path.is_file())

            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["sample_count"], 1)
            self.assertEqual(summary["input_file_count"], 1)
            self.assertEqual(summary["metric_names"], ["patch_error"])

            with records_csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_id"], "sample")
            self.assertEqual(rows[0]["metric"], "patch_error")


if __name__ == "__main__":
    unittest.main()
