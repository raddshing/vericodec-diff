from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vericodec_diff.real_images import default_real_image_config, prepare_real_images, resolve_real_image_config
from vericodec_diff.reconstruction_records import (
    build_reconstruction_records,
    default_reconstruction_record_config,
    resolve_reconstruction_record_config,
    validate_reconstruction_records,
)
from vericodec_diff.synthetic_dataset import default_generation_config, generate_dataset, resolve_generation_config


SCRIPT_PATH = REPO_ROOT / "scripts" / "build_reconstruction_records.py"


def _write_source_image(path: Path, *, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    draw.ellipse((size[0] // 4, size[1] // 4, size[0] - (size[0] // 4), size[1] - (size[1] // 4)), outline=(255, 255, 255), width=6)
    image.save(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _prepare_input_manifests(repo_root: Path, *, synthetic_count: int, real_count: int) -> tuple[Path, Path]:
    synthetic_raw_config = default_generation_config()
    synthetic_raw_config["paths"]["repo_root"] = str(repo_root)
    synthetic_raw_config["dataset"]["splits"] = ["kill"]
    synthetic_raw_config["dataset"]["categories"] = ["text-posters-signs"]
    synthetic_raw_config["dataset"]["split_counts"]["kill"] = synthetic_count
    synthetic_raw_config["dataset"]["split_counts"]["main"] = 0
    synthetic_raw_config["dataset"]["split_counts"]["hard"] = 0
    synthetic_config = resolve_generation_config(repo_root, synthetic_raw_config)
    generate_dataset(synthetic_config)

    ffhq_root = repo_root / "sources" / "ffhq"
    for index in range(real_count):
        width = 1200 + (index * 20)
        height = 1000 + (index * 30)
        _write_source_image(
            ffhq_root / f"face_{index:03d}.jpg",
            size=(width, height),
            color=(50 + index * 10, 90 + index * 5, 140 - index * 5),
        )

    real_raw_config = default_real_image_config()
    real_raw_config["paths"]["repo_root"] = str(repo_root)
    real_raw_config["source"].update({"kind": "ffhq", "root": str(ffhq_root)})
    real_raw_config["selection"].update({"category": "faces", "split": "real", "limit": real_count})
    real_config = resolve_real_image_config(repo_root, real_raw_config)
    prepare_real_images(real_config)

    synthetic_manifest_path = Path(synthetic_config["paths"]["manifest_path"])
    real_manifest_path = Path(real_config["paths"]["manifest_path"])
    return synthetic_manifest_path, real_manifest_path


class ReconstructionRecordTests(unittest.TestCase):
    def test_build_reconstruction_records_is_deterministic_and_stratified(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            synthetic_manifest_path, real_manifest_path = _prepare_input_manifests(
                repo_root,
                synthetic_count=5,
                real_count=5,
            )

            raw_config = default_reconstruction_record_config()
            raw_config["paths"].update(
                {
                    "repo_root": str(repo_root),
                    "synthetic_manifest_path": str(synthetic_manifest_path),
                    "real_manifest_path": str(real_manifest_path),
                }
            )
            config = resolve_reconstruction_record_config(repo_root, raw_config)

            first_summary = build_reconstruction_records(config)
            records_path = Path(config["paths"]["records_path"])
            first_csv = records_path.read_text(encoding="utf-8")

            second_summary = build_reconstruction_records(config)
            second_csv = records_path.read_text(encoding="utf-8")

            self.assertEqual(first_summary["record_count"], 10)
            self.assertEqual(second_summary["split_counts"], {"test": 2, "train": 6, "val": 2})
            self.assertEqual(first_csv, second_csv)

            summary = validate_reconstruction_records(records_path, repo_root=repo_root, verify_files=True)
            self.assertEqual(summary["category_counts"], {"faces": 5, "text-posters-signs": 5})
            self.assertEqual(summary["split_counts"], {"test": 2, "train": 6, "val": 2})

            per_category_split_counts = Counter((row["category"], row["split"]) for row in _read_rows(records_path))
            self.assertEqual(per_category_split_counts[("faces", "train")], 3)
            self.assertEqual(per_category_split_counts[("faces", "val")], 1)
            self.assertEqual(per_category_split_counts[("faces", "test")], 1)
            self.assertEqual(per_category_split_counts[("text-posters-signs", "train")], 3)
            self.assertEqual(per_category_split_counts[("text-posters-signs", "val")], 1)
            self.assertEqual(per_category_split_counts[("text-posters-signs", "test")], 1)

    def test_build_reconstruction_records_script_writes_resolved_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            synthetic_manifest_path, real_manifest_path = _prepare_input_manifests(
                repo_root,
                synthetic_count=5,
                real_count=5,
            )

            command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--repo-root",
                tmpdir,
                "--synthetic-manifest-path",
                str(synthetic_manifest_path),
                "--real-manifest-path",
                str(real_manifest_path),
            ]
            subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            resolved_config_path = repo_root / "outputs" / "reconstruction_records" / "resolved_config.yaml"
            records_path = repo_root / "data" / "manifests" / "reconstruction_records.csv"

            self.assertTrue(resolved_config_path.is_file())
            self.assertTrue(records_path.is_file())

            summary = validate_reconstruction_records(records_path, repo_root=repo_root, verify_files=True)
            self.assertEqual(summary["total"], 10)


if __name__ == "__main__":
    unittest.main()
