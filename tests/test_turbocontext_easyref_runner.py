from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = REPO_ROOT / "scripts" / "run_easyref_baseline.py"
PROFILE_SCRIPT = REPO_ROOT / "scripts" / "profile_refmem_vram.py"


def _write_reference_images(repo_root: Path) -> list[str]:
    reference_dir = repo_root / "refs"
    reference_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index in range(8):
        path = reference_dir / f"ref_{index + 1:02d}.png"
        Image.new(
            "RGB",
            (1024, 1024),
            color=((index * 31) % 255, (index * 47) % 255, (index * 59) % 255),
        ).save(path)
        paths.append(display_path(path, repo_root))
    return paths


def display_path(path: Path, repo_root: Path) -> str:
    return str(path.relative_to(repo_root))


def _write_config(repo_root: Path) -> Path:
    config = {
        "paths": {
            "repo_root": str(repo_root),
            "suite": "test_easyref_suite",
        },
        "backend": {
            "kind": "mock",
        },
        "generation": {
            "ref_count": 4,
        },
        "profiling": {
            "ref_counts": [1, 2, 4, 8],
        },
        "suite": {
            "items": [
                {
                    "prompt_id": "smoke_easyref_001",
                    "prompt": "A deterministic portrait study.",
                    "negative_prompt": "bad anatomy, low quality",
                    "seed": 12345,
                    "reference_images": _write_reference_images(repo_root),
                }
            ]
        },
    }
    config_path = repo_root / "easyref_test_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path


class EasyRefRunnerScriptTests(unittest.TestCase):
    def test_run_script_writes_smoke_outputs_and_resolved_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            config_path = _write_config(repo_root)

            subprocess.run(
                [sys.executable, str(RUN_SCRIPT), "--config", str(config_path)],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            output_dir = repo_root / "outputs" / "baselines" / "easyref" / "test_easyref_suite" / "ref_count_4"
            self.assertTrue((output_dir / "resolved_config.yaml").is_file())
            self.assertTrue((output_dir / "easyref_run_summary.json").is_file())
            self.assertTrue((output_dir / "easyref_run_records.csv").is_file())
            self.assertTrue((output_dir / "smoke_easyref_001__seed_12345.png").is_file())

            with (output_dir / "easyref_run_summary.json").open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["generated_sample_count"], 1)
            self.assertEqual(summary["ref_count"], 4)

            with (output_dir / "easyref_run_records.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prompt_id"], "smoke_easyref_001")
            self.assertEqual(rows[0]["ref_count"], "4")

    def test_profile_script_writes_metric_schema_and_component_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            config_path = _write_config(repo_root)

            subprocess.run(
                [sys.executable, str(PROFILE_SCRIPT), "--config", str(config_path)],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            output_dir = repo_root / "outputs" / "metrics" / "turbocontext_gate"
            resolved_config_path = output_dir / "resolved_config.yaml"
            summary_path = output_dir / "easyref_refmem_profile.json"
            records_csv_path = output_dir / "easyref_refmem_profile.csv"

            self.assertTrue(resolved_config_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertTrue(records_csv_path.is_file())

            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            self.assertEqual(summary["run_count"], 4)
            self.assertEqual(summary["measurement_modes"], ["mock_direct_tensor_bytes"])
            self.assertEqual(
                summary["component_names"],
                [
                    "cond_reference_encode",
                    "diffusion_decode",
                    "prompt_conditioning",
                    "uncond_reference_encode",
                ],
            )
            self.assertEqual(len(summary["records"]), 4)
            self.assertIn("component_peaks_gb", summary["records"][0])
            self.assertIn("refmem_components_gb", summary["records"][0])

            with records_csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(
                    tuple(reader.fieldnames or ()),
                    (
                        "prompt_id",
                        "seed",
                        "ref_count",
                        "total_peak_vram_gb",
                        "refmem_peak_vram_gb",
                        "refmem_share",
                    ),
                )
            self.assertEqual(len(rows), 4)
            self.assertEqual([row["ref_count"] for row in rows], ["1", "2", "4", "8"])
            for row in rows:
                total = float(row["total_peak_vram_gb"])
                refmem = float(row["refmem_peak_vram_gb"])
                share = float(row["refmem_share"])
                self.assertGreater(total, 0.0)
                self.assertGreater(refmem, 0.0)
                self.assertAlmostEqual(share, refmem / total, places=6)


if __name__ == "__main__":
    unittest.main()
