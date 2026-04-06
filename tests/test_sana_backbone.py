from __future__ import annotations

import csv
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vericodec_diff.prompt_manifests import REQUIRED_COLUMNS
from vericodec_diff.sana_backbone import (
    build_sample_id,
    generate_sana_outputs,
    resolve_sana_generation_config,
    write_generation_records_csv,
)


class _FakeTorch:
    @staticmethod
    def save(payload: object, path: str | Path) -> None:
        with Path(path).open("wb") as handle:
            pickle.dump(payload, handle)


class _FakeBackbone:
    def __init__(self) -> None:
        self.seeds: list[int] = []

    def generate_sample(
        self,
        *,
        sample_id: str,
        prompt_id: str,
        prompt: str,
        negative_prompt: str,
        seed: int,
        trace_config: dict[str, object],
    ) -> dict[str, object]:
        del prompt_id, prompt, negative_prompt
        self.seeds.append(seed)
        image = Image.new("RGB", (1024, 1024), color=(seed % 255, 64, 128))

        trace_payload = None
        trace_status = "disabled"
        limitations: list[str] = []
        if trace_config["save_trace"]:
            trace_status = "final_only"
            trace_payload = {
                "sample_id": sample_id,
                "trace_status": trace_status,
                "png_bytes": [b"fake-trace"],
            }
            limitations.append(
                "Backend does not expose callback_on_step_end; late-step traces fall back to final latent only."
            )

        return {
            "image": image,
            "trace_payload": trace_payload,
            "trace_status": trace_status,
            "backend_limitations": limitations,
        }


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "prompt_id": "smoke_text_heavy_001",
            "category": "text-heavy",
            "stress_type": "dense_multiline_copy",
            "prompt": 'poster that says "HELLO"',
            "negative_prompt": "blurry text",
            "seed": "123",
            "split": "smoke",
        },
        {
            "prompt_id": "smoke_faces_002",
            "category": "faces",
            "stress_type": "eyelashes",
            "prompt": "portrait photo with sharp eyelashes",
            "negative_prompt": "oversmoothed skin",
            "seed": "456",
            "split": "smoke",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class SanaBackboneTests(unittest.TestCase):
    def test_build_sample_id_uses_prompt_id_and_seed(self) -> None:
        self.assertEqual(build_sample_id("kill_text_heavy_001", 110001), "kill_text_heavy_001__seed_110001")

    def test_resolve_config_derives_suite_from_manifest_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            manifest_path = repo_root / "data" / "manifests" / "smoke_manifest.csv"
            _write_manifest(manifest_path)

            config = resolve_sana_generation_config(
                repo_root,
                {
                    "paths": {
                        "repo_root": ".",
                        "prompt_manifest": "data/manifests/smoke_manifest.csv",
                    }
                },
            )

            self.assertEqual(config["paths"]["suite"], "smoke_manifest")
            self.assertEqual(
                Path(config["paths"]["output_dir"]),
                repo_root / "outputs" / "sana" / "smoke_manifest",
            )

    def test_generate_outputs_writes_pngs_trace_files_and_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            manifest_path = repo_root / "data" / "manifests" / "smoke_manifest.csv"
            _write_manifest(manifest_path)

            config = resolve_sana_generation_config(
                repo_root,
                {
                    "paths": {
                        "repo_root": ".",
                        "prompt_manifest": "data/manifests/smoke_manifest.csv",
                        "suite": "smoke",
                    },
                    "trace": {
                        "save_trace": True,
                        "late_step_count": 2,
                    },
                },
            )

            backbone = _FakeBackbone()
            result = generate_sana_outputs(config, backbone=backbone, torch_module=_FakeTorch())
            output_dir = Path(config["paths"]["output_dir"])

            first_sample = output_dir / "smoke_text_heavy_001__seed_123.png"
            first_trace = output_dir / "smoke_text_heavy_001__seed_123__trace.pt"
            second_sample = output_dir / "smoke_faces_002__seed_456.png"
            second_trace = output_dir / "smoke_faces_002__seed_456__trace.pt"

            self.assertTrue(first_sample.is_file())
            self.assertTrue(first_trace.is_file())
            self.assertTrue(second_sample.is_file())
            self.assertTrue(second_trace.is_file())

            summary = result["summary"]
            self.assertEqual(summary["generated_sample_count"], 2)
            self.assertEqual(summary["trace"]["saved_count"], 2)
            self.assertEqual(summary["trace"]["status_counts"], {"final_only": 2})
            self.assertEqual(backbone.seeds, [123, 456])

            with first_trace.open("rb") as handle:
                payload = pickle.load(handle)
            self.assertEqual(payload["sample_id"], "smoke_text_heavy_001__seed_123")

            records_csv_path = output_dir / "generation_records.csv"
            write_generation_records_csv(records_csv_path, result["records"])
            with records_csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["trace_status"], "final_only")


if __name__ == "__main__":
    unittest.main()
