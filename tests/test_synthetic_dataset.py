from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vericodec_diff.synthetic_dataset import (
    default_generation_config,
    generate_dataset,
    resolve_generation_config,
    validate_synthetic_manifest,
)


SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_diagnostic_dataset.py"


def _small_config(repo_root: Path) -> dict[str, object]:
    raw_config = default_generation_config()
    raw_config["paths"]["repo_root"] = str(repo_root)
    raw_config["dataset"]["splits"] = ["kill"]
    raw_config["dataset"]["split_counts"]["kill"] = 1
    raw_config["dataset"]["split_counts"]["main"] = 0
    raw_config["dataset"]["split_counts"]["hard"] = 0
    return resolve_generation_config(repo_root, raw_config)


class SyntheticDatasetTests(unittest.TestCase):
    def test_generated_images_are_exactly_1024_square(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _small_config(Path(tmpdir))
            summary = generate_dataset(config)
            self.assertEqual(summary["image_count"], 3)

            manifest_path = Path(config["paths"]["manifest_path"])
            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            for row in rows:
                image_path = Path(config["paths"]["repo_root"]) / row["local_path"]
                with Image.open(image_path) as image:
                    self.assertEqual(image.size, (1024, 1024))

    def test_manifest_integrity_and_resolved_config_from_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--repo-root",
                tmpdir,
                "--splits",
                "kill",
                "--count-override",
                "kill=1",
            ]
            subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            resolved_config_path = Path(tmpdir) / "outputs" / "diagnostic_dataset" / "resolved_config.yaml"
            manifest_path = Path(tmpdir) / "data" / "manifests" / "synthetic_manifest.csv"
            self.assertTrue(resolved_config_path.is_file())
            self.assertTrue(manifest_path.is_file())

            config = _small_config(Path(tmpdir))
            summary = validate_synthetic_manifest(
                manifest_path,
                repo_root=Path(config["paths"]["repo_root"]),
                config=config,
                verify_files=True,
            )
            self.assertEqual(summary["total"], 3)
            self.assertEqual(summary["split_counts"], {"kill": 3})

    def test_generation_is_deterministic_for_same_config_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as first_tmpdir, tempfile.TemporaryDirectory() as second_tmpdir:
            first_config = _small_config(Path(first_tmpdir))
            second_config = _small_config(Path(second_tmpdir))

            generate_dataset(first_config)
            generate_dataset(second_config)

            first_manifest = Path(first_config["paths"]["manifest_path"])
            second_manifest = Path(second_config["paths"]["manifest_path"])
            self.assertEqual(first_manifest.read_text(encoding="utf-8"), second_manifest.read_text(encoding="utf-8"))

            with first_manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            for row in rows:
                first_image = Path(first_config["paths"]["repo_root"]) / row["local_path"]
                second_image = Path(second_config["paths"]["repo_root"]) / row["local_path"]
                first_digest = hashlib.sha256(first_image.read_bytes()).hexdigest()
                second_digest = hashlib.sha256(second_image.read_bytes()).hexdigest()
                self.assertEqual(first_digest, second_digest)


if __name__ == "__main__":
    unittest.main()
