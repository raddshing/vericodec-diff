from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vericodec_diff.patch_metrics import (
    PixelL2LpipsScorer,
    compute_patch_error_arrays,
)


SCRIPT_PATH = REPO_ROOT / "scripts" / "compute_patch_errors.py"


def _checkerboard_patch(size: int = 64, cell: int = 1) -> np.ndarray:
    grid_y, grid_x = np.indices((size, size))
    pattern = ((grid_x // cell) + (grid_y // cell)) % 2
    return (pattern * 255).astype(np.uint8)


def _pattern_image() -> Image.Image:
    array = np.zeros((1024, 1024, 3), dtype=np.uint8)
    patch = _checkerboard_patch()
    array[:64, :64, 0] = patch
    array[:64, :64, 1] = patch
    array[:64, :64, 2] = patch
    return Image.fromarray(array, mode="RGB")


def _blank_image() -> Image.Image:
    return Image.new("RGB", (1024, 1024), color=(0, 0, 0))


class PatchMetricTests(unittest.TestCase):
    def test_patch_metrics_return_256_finite_values_and_localize_the_changed_patch(self) -> None:
        source_image = _pattern_image()
        reconstruction_image = _blank_image()
        metric_arrays = compute_patch_error_arrays(
            source_image,
            reconstruction_image,
            lpips_scorer=PixelL2LpipsScorer(),
            wavelet="haar",
        )

        for metric_name, values in metric_arrays.items():
            with self.subTest(metric=metric_name):
                self.assertEqual(values.shape, (256,))
                self.assertTrue(np.all(np.isfinite(values)))
                self.assertGreater(float(values[0]), 0.0)
                self.assertEqual(int(np.count_nonzero(values > 1e-8)), 1)

    def test_compute_patch_errors_script_writes_npz_preview_and_resolved_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            source_path = repo_root / "data" / "source" / "sample.png"
            reconstruction_path = repo_root / "data" / "recon" / "sample.png"
            records_csv_path = repo_root / "data" / "records.csv"

            source_path.parent.mkdir(parents=True, exist_ok=True)
            reconstruction_path.parent.mkdir(parents=True, exist_ok=True)
            _pattern_image().save(source_path)
            _blank_image().save(reconstruction_path)

            with records_csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("sample_id", "split", "image_path", "reconstruction_path"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sample_id": "sample",
                        "split": "kill",
                        "image_path": "data/source/sample.png",
                        "reconstruction_path": "data/recon/sample.png",
                    }
                )

            command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--repo-root",
                tmpdir,
                "--records-csv",
                "data/records.csv",
                "--reconstruction-mode",
                "precomputed",
                "--lpips-backend",
                "pixel_l2",
            ]
            subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            npz_path = repo_root / "outputs" / "error_maps" / "kill" / "sample__patch64.npz"
            preview_path = repo_root / "outputs" / "error_maps" / "kill" / "sample__patch64_preview.png"
            resolved_config_path = repo_root / "outputs" / "error_maps" / "resolved_config.yaml"
            summary_path = repo_root / "outputs" / "error_maps" / "patch_error_summary.json"
            records_output_csv = repo_root / "outputs" / "error_maps" / "patch_error_records.csv"

            self.assertTrue(npz_path.is_file())
            self.assertTrue(preview_path.is_file())
            self.assertTrue(resolved_config_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertTrue(records_output_csv.is_file())

            with np.load(npz_path, allow_pickle=False) as payload:
                self.assertEqual(payload["lpips"].shape, (256,))
                self.assertEqual(payload["one_minus_ssim"].shape, (256,))
                self.assertEqual(payload["hf_wavelet_l1"].shape, (256,))


if __name__ == "__main__":
    unittest.main()
