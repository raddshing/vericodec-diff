from __future__ import annotations

import csv
import io
import json
import pickle
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

from vericodec_diff.verifier_features import (
    extract_verifier_features,
    resolve_verifier_feature_config,
)


SCRIPT_PATH = REPO_ROOT / "scripts" / "extract_verifier_signals.py"
IMPLEMENTED_SIGNAL_NAMES = (
    "decode_reencode_consistency",
    "late_step_update_magnitude",
    "wavelet_instability",
    "sobel_edge_magnitude",
    "local_intensity_variance",
    "patch_entropy",
)


def _checkerboard_patch(size: int = 64, cell: int = 4) -> np.ndarray:
    grid_y, grid_x = np.indices((size, size))
    pattern = ((grid_x // cell) + (grid_y // cell)) % 2
    return (pattern * 255).astype(np.uint8)


def _pattern_image(scale: float = 1.0) -> Image.Image:
    array = np.zeros((1024, 1024, 3), dtype=np.uint8)
    patch = np.rint(_checkerboard_patch().astype(np.float32) * scale).clip(0, 255).astype(np.uint8)
    array[:64, :64, :] = patch[:, :, None]
    array[64:128, 64:128, 0] = patch
    array[128:192, 128:192, 1] = patch
    array[192:256, 192:256, 2] = patch

    gradient = np.linspace(0, 255, 64, dtype=np.uint8)
    array[256:320, :64, 0] = gradient[None, :]
    array[256:320, :64, 1] = gradient[::-1][None, :]
    array[256:320, :64, 2] = gradient[None, :]
    return Image.fromarray(array, mode="RGB")


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_trace(path: Path) -> None:
    trace_payload = {
        "format": "png_bytes",
        "trace_status": "late_steps",
        "step_indices": [17, 18, 19],
        "png_bytes": [
            _png_bytes(Image.new("RGB", (1024, 1024), color=(0, 0, 0))),
            _png_bytes(_pattern_image(scale=0.5)),
            _png_bytes(_pattern_image(scale=1.0)),
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(trace_payload, handle)


def _write_records_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "split",
                "prompt_id",
                "category",
                "stress_type",
                "seed",
                "image_path",
                "trace_path",
                "trace_status",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample",
                "split": "smoke",
                "prompt_id": "smoke_001",
                "category": "text-heavy",
                "stress_type": "checkerboard",
                "seed": "123",
                "image_path": "data/generated/sample.png",
                "trace_path": "data/traces/sample__trace.pt",
                "trace_status": "late_steps",
            }
        )


def _build_temp_repo(root: Path) -> None:
    image_path = root / "data" / "generated" / "sample.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    _pattern_image().save(image_path)
    _write_trace(root / "data" / "traces" / "sample__trace.pt")
    _write_records_csv(root / "data" / "records.csv")


class VerifierFeatureTests(unittest.TestCase):
    def test_extract_verifier_features_writes_256_finite_vectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            _build_temp_repo(repo_root)

            config = resolve_verifier_feature_config(
                repo_root,
                {
                    "paths": {
                        "repo_root": ".",
                        "records_csv": "data/records.csv",
                    },
                    "decode_reencode": {
                        "backend": "jpeg",
                        "jpeg_quality": 90,
                    },
                },
            )
            result = extract_verifier_features(config)

            self.assertEqual(result["summary"]["generated_feature_count"], 1)
            npz_path = repo_root / "data" / "processed" / "verifier_features" / "smoke" / "sample__signals.npz"
            index_path = repo_root / "data" / "processed" / "verifier_features" / "smoke" / "index.csv"
            self.assertTrue(npz_path.is_file())
            self.assertTrue(index_path.is_file())

            with np.load(npz_path, allow_pickle=False) as payload:
                for signal_name in IMPLEMENTED_SIGNAL_NAMES:
                    with self.subTest(signal=signal_name):
                        self.assertIn(signal_name, payload.files)
                        values = payload[signal_name]
                        self.assertEqual(values.shape, (256,))
                        self.assertTrue(np.all(np.isfinite(values)))
                self.assertEqual(payload["missing_signal_order"].tolist(), [])

    def test_script_reports_optional_cross_attention_signal_as_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            _build_temp_repo(repo_root)

            command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--repo-root",
                tmpdir,
                "--records-csv",
                "data/records.csv",
                "--decode-reencode-backend",
                "jpeg",
                "--enable-cross-attention-instability",
            ]
            subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            feature_root = repo_root / "data" / "processed" / "verifier_features"
            npz_path = feature_root / "smoke" / "sample__signals.npz"
            index_path = feature_root / "smoke" / "index.csv"
            resolved_config_path = feature_root / "resolved_config.yaml"
            summary_path = feature_root / "verifier_feature_summary.json"

            self.assertTrue(npz_path.is_file())
            self.assertTrue(index_path.is_file())
            self.assertTrue(resolved_config_path.is_file())
            self.assertTrue(summary_path.is_file())

            with np.load(npz_path, allow_pickle=False) as payload:
                self.assertNotIn("cross_attention_instability", payload.files)
                self.assertEqual(payload["missing_signal_order"].tolist(), ["cross_attention_instability"])
                missing_reasons = json.loads(str(payload["missing_signal_reasons_json"].item()))
                self.assertIn("cross_attention_instability", missing_reasons)

            with index_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["missing_signals"], "cross_attention_instability")

            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["missing_optional_signal_counts"], {"cross_attention_instability": 1})


if __name__ == "__main__":
    unittest.main()
