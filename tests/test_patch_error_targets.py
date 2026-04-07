from __future__ import annotations

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

from vericodec_diff.patch_error_targets import (
    PATCH_ERROR_METRIC_NAME,
    PATCH_LABEL_TOP15_NAME,
    PATCH_LABEL_TOP15_PERCENT,
    augment_patch_error_payload,
)
from vericodec_diff.patch_metrics import SampleRecord, build_patch_error_payload, write_patch_error_npz


SCRIPT_PATH = REPO_ROOT / "scripts" / "augment_patch_error_targets.py"


def _base_payload(metric_values: np.ndarray) -> dict[str, np.ndarray]:
    record = SampleRecord(
        sample_id="sample",
        split="kill",
        source_path=Path("data/source/sample.png"),
        reconstruction_path=Path("data/recon/sample.png"),
        source_local_path="data/source/sample.png",
        reconstruction_local_path="data/recon/sample.png",
        metadata={},
    )
    return build_patch_error_payload(
        record,
        {
            "lpips": metric_values,
            "one_minus_ssim": metric_values * np.float32(0.2),
            "hf_wavelet_l1": metric_values * np.float32(0.4),
        },
        lpips_backend="pixel_l2_debug",
        wavelet="haar",
    )


class PatchErrorTargetTests(unittest.TestCase):
    def test_augment_patch_error_payload_creates_weighted_metric_and_top15_mask(self) -> None:
        metric_values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        augmented = augment_patch_error_payload(_base_payload(metric_values))

        expected_patch_error = (0.5 * metric_values + 0.25 * (metric_values * 0.2) + 0.25 * (metric_values * 0.4)).astype(
            np.float32
        )
        np.testing.assert_allclose(augmented[PATCH_ERROR_METRIC_NAME], expected_patch_error, atol=1e-7, rtol=0.0)

        label_values = np.asarray(augmented[PATCH_LABEL_TOP15_NAME], dtype=np.uint8)
        expected_positive_count = int(np.ceil(256 * (PATCH_LABEL_TOP15_PERCENT / 100.0)))
        self.assertEqual(int(label_values.sum(dtype=np.int64)), expected_positive_count)
        np.testing.assert_array_equal(
            np.flatnonzero(label_values),
            np.arange(256 - expected_positive_count, 256, dtype=np.int64),
        )

    def test_augment_patch_error_targets_script_rewrites_npz_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            error_map_path = repo_root / "outputs" / "error_maps" / "kill" / "sample__patch64.npz"
            metric_values = np.linspace(0.0, 1.0, 256, dtype=np.float32)
            write_patch_error_npz(error_map_path, _base_payload(metric_values))

            command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--repo-root",
                tmpdir,
                "--split-filter",
                "kill",
                "--limit",
                "1",
            ]
            subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            output_dir = repo_root / "outputs" / "metrics" / "kill_test" / "patch_error_targets"
            summary_path = output_dir / "patch_error_targets_summary.json"
            self.assertTrue((output_dir / "resolved_config.yaml").is_file())
            self.assertTrue((output_dir / "patch_error_target_records.csv").is_file())
            self.assertTrue(summary_path.is_file())

            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["continuous_metric_name"], PATCH_ERROR_METRIC_NAME)
            self.assertEqual(summary["label_metric_name"], PATCH_LABEL_TOP15_NAME)
            self.assertEqual(summary["input_file_count"], 1)
            self.assertEqual(summary["updated_file_count"], 1)

            with np.load(error_map_path, allow_pickle=False) as payload:
                self.assertIn(PATCH_ERROR_METRIC_NAME, payload.files)
                self.assertIn(PATCH_LABEL_TOP15_NAME, payload.files)


if __name__ == "__main__":
    unittest.main()
