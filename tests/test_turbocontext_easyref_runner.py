from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

RUN_SCRIPT = REPO_ROOT / "scripts" / "run_easyref_baseline.py"
PROFILE_SCRIPT = REPO_ROOT / "scripts" / "profile_refmem_vram.py"
ATTENTION_PROCESSOR_PATH = (
    REPO_ROOT
    / "baselines"
    / "external"
    / "templex98_easyref"
    / "ip_adapter"
    / "attention_processor.py"
)

from turbocontext.easyref_runner import GB, build_reference_interface_metrics


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
            self.assertIn("persistent_ref_interface_bytes", summary["records"][0])
            self.assertIn("reference_encode_peak_delta_gb", summary["records"][0])

            with records_csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(
                    tuple(reader.fieldnames or ()),
                    (
                        "prompt_id",
                        "seed",
                        "ref_count",
                        "ref_tokens_cond_bytes",
                        "ref_tokens_uncond_bytes",
                        "persistent_ref_interface_bytes",
                        "persistent_ref_interface_gb",
                        "persistent_ref_interface_share",
                        "reference_encode_peak_delta_gb",
                        "reference_encode_peak_vram_gb",
                        "total_peak_vram_gb",
                    ),
                )
            self.assertEqual(len(rows), 4)
            self.assertEqual([row["ref_count"] for row in rows], ["1", "2", "4", "8"])
            for row in rows:
                cond_bytes = int(row["ref_tokens_cond_bytes"])
                uncond_bytes = int(row["ref_tokens_uncond_bytes"])
                persistent_bytes = int(row["persistent_ref_interface_bytes"])
                persistent_gb = float(row["persistent_ref_interface_gb"])
                persistent_share = float(row["persistent_ref_interface_share"])
                reference_encode_peak_delta_gb = float(row["reference_encode_peak_delta_gb"])
                reference_encode_peak_vram_gb = float(row["reference_encode_peak_vram_gb"])
                total = float(row["total_peak_vram_gb"])
                self.assertGreater(total, 0.0)
                self.assertGreater(cond_bytes, 0)
                self.assertEqual(cond_bytes, uncond_bytes)
                self.assertEqual(persistent_bytes, cond_bytes + uncond_bytes)
                self.assertAlmostEqual(persistent_gb, persistent_bytes / GB, places=6)
                self.assertAlmostEqual(persistent_share, persistent_gb / total, places=6)
                self.assertGreater(reference_encode_peak_delta_gb, 0.0)
                self.assertGreater(reference_encode_peak_vram_gb, 0.0)

    def test_build_reference_interface_metrics_splits_persistent_bytes_from_encode_peaks(self) -> None:
        metrics = build_reference_interface_metrics(
            ref_tokens_cond_bytes=262_144,
            ref_tokens_uncond_bytes=262_144,
            total_peak_vram_gb=3.08,
            cond_reference_peak={"peak_delta_vram_gb": 0.63, "peak_vram_gb": 2.38},
            uncond_reference_peak={"peak_delta_vram_gb": 0.25, "peak_vram_gb": 2.11},
        )

        self.assertEqual(metrics["ref_tokens_cond_bytes"], 262_144)
        self.assertEqual(metrics["ref_tokens_uncond_bytes"], 262_144)
        self.assertEqual(metrics["persistent_ref_interface_bytes"], 524_288)
        self.assertAlmostEqual(
            float(metrics["persistent_ref_interface_gb"]),
            524_288 / GB,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["persistent_ref_interface_share"]),
            float(metrics["persistent_ref_interface_gb"]) / 3.08,
            places=6,
        )
        self.assertEqual(metrics["reference_encode_peak_delta_gb"], 0.63)
        self.assertEqual(metrics["reference_encode_peak_vram_gb"], 2.38)

    def test_ip_attention_verification_hook_records_transient_materialization(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "easyref_attention_processor_test",
            ATTENTION_PROCESSOR_PATH,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class FakeAttention(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.spatial_norm = None
                self.group_norm = None
                self.norm_cross = False
                self.heads = 2
                self.residual_connection = False
                self.rescale_output_factor = 1.0
                self.to_q = torch.nn.Linear(4, 4, bias=False)
                self.to_k = torch.nn.Linear(4, 4, bias=False)
                self.to_v = torch.nn.Linear(4, 4, bias=False)
                self.to_out = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

        processor = module.IPAttnProcessor2_0(
            hidden_size=4,
            cross_attention_dim=4,
            num_tokens=2,
        )
        captured: list[dict[str, object]] = []
        processor.ip_attention_verification_hook = captured.append

        output = processor(
            FakeAttention(),
            torch.randn(1, 3, 4),
            encoder_hidden_states=torch.randn(1, 5, 4),
        )

        self.assertEqual(tuple(output.shape), (1, 3, 4))
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["num_tokens"], 2)
        self.assertEqual(captured[0]["ip_hidden_states_shape"], [1, 2, 4])
        self.assertEqual(captured[0]["ip_kv_materialization"], "transient_in_call")


if __name__ == "__main__":
    unittest.main()
