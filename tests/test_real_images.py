from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vericodec_diff.real_images import (
    default_real_image_config,
    prepare_real_images,
    resolve_real_image_config,
    validate_real_manifest,
)


SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_real_images.py"


def _write_source_image(path: Path, *, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((size[0] // 5, size[1] // 5, size[0] - (size[0] // 5), size[1] - (size[1] // 5)), outline=(255, 255, 255), width=8)
    image.save(path)


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class RealImagePreparationTests(unittest.TestCase):
    def test_prepare_real_images_from_ffhq_is_deterministic_and_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            source_root = repo_root / "local_sources" / "ffhq"
            _write_source_image(source_root / "set_a" / "face_001.jpg", size=(1600, 1200), color=(120, 40, 70))
            _write_source_image(source_root / "set_b" / "face_002.png", size=(900, 1500), color=(30, 120, 170))

            raw_config = default_real_image_config()
            raw_config["paths"]["repo_root"] = str(repo_root)
            raw_config["source"].update({"kind": "ffhq", "root": str(source_root)})
            raw_config["selection"].update({"category": "faces", "split": "curated"})
            config = resolve_real_image_config(repo_root, raw_config)

            first_summary = prepare_real_images(config)
            manifest_path = Path(config["paths"]["manifest_path"])
            first_manifest_text = manifest_path.read_text(encoding="utf-8")

            second_summary = prepare_real_images(config)
            second_manifest_text = manifest_path.read_text(encoding="utf-8")

            self.assertEqual(first_summary["prepared_count"], 2)
            self.assertEqual(second_summary["manifest_count"], 2)
            self.assertEqual(first_manifest_text, second_manifest_text)

            summary = validate_real_manifest(manifest_path, repo_root=repo_root, verify_files=True)
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["category_counts"], {"faces": 2})
            self.assertEqual(summary["split_counts"], {"curated": 2})

            for row in _read_manifest_rows(manifest_path):
                image_path = repo_root / row["local_path"]
                with Image.open(image_path) as image:
                    self.assertEqual(image.size, (1024, 1024))

    def test_prepare_real_images_script_appends_openimages_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            ffhq_root = repo_root / "sources" / "ffhq"
            openimages_root = repo_root / "sources" / "openimages"

            _write_source_image(ffhq_root / "0001.jpg", size=(1400, 1000), color=(50, 90, 150))
            _write_source_image(ffhq_root / "0002.jpg", size=(1000, 1400), color=(90, 150, 50))
            _write_source_image(openimages_root / "dog" / "dog_001.jpg", size=(1300, 900), color=(160, 90, 40))
            _write_source_image(openimages_root / "dog" / "dog_002.jpg", size=(900, 1300), color=(40, 160, 90))
            _write_source_image(openimages_root / "cat" / "cat_001.jpg", size=(1200, 1200), color=(90, 40, 160))

            ffhq_command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--repo-root",
                tmpdir,
                "--ffhq-root",
                str(ffhq_root),
                "--category",
                "faces",
                "--split",
                "real",
                "--limit",
                "2",
            ]
            subprocess.run(ffhq_command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            openimages_command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--repo-root",
                tmpdir,
                "--openimages-root",
                str(openimages_root),
                "--openimages-class",
                "dog",
                "--category",
                "animals",
                "--split",
                "real",
                "--limit",
                "2",
            ]
            subprocess.run(openimages_command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            manifest_path = repo_root / "data" / "manifests" / "real_images_manifest.csv"
            resolved_config_path = repo_root / "outputs" / "real_image_prep" / "resolved_config.yaml"
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(resolved_config_path.is_file())

            rows = _read_manifest_rows(manifest_path)
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["category"] for row in rows}, {"faces", "animals"})
            self.assertTrue(all("cat" not in row["local_path"] for row in rows if row["category"] == "animals"))

            summary = validate_real_manifest(manifest_path, repo_root=repo_root, verify_files=True)
            self.assertEqual(summary["category_counts"], {"animals": 2, "faces": 2})

    def test_prepare_real_images_from_huggingface_imagefolder(self) -> None:
        try:
            import datasets  # noqa: F401
        except ImportError:
            self.skipTest("datasets is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            data_dir = repo_root / "hf_sources" / "imagefolder"
            _write_source_image(data_dir / "sample_class" / "img_001.jpg", size=(1500, 1100), color=(180, 60, 60))
            _write_source_image(data_dir / "sample_class" / "img_002.jpg", size=(1100, 1500), color=(60, 60, 180))

            raw_config = default_real_image_config()
            raw_config["paths"]["repo_root"] = str(repo_root)
            raw_config["source"].update(
                {
                    "kind": "huggingface",
                    "hf_dataset": "imagefolder",
                    "hf_split": "train",
                    "hf_data_dir": str(data_dir),
                }
            )
            raw_config["selection"].update({"category": "hf-samples", "limit": 2})
            config = resolve_real_image_config(repo_root, raw_config)

            summary = prepare_real_images(config)
            manifest_path = Path(config["paths"]["manifest_path"])

            self.assertEqual(summary["prepared_count"], 2)
            manifest_summary = validate_real_manifest(manifest_path, repo_root=repo_root, verify_files=True)
            self.assertEqual(manifest_summary["source_counts"], {"imagefolder": 2})


if __name__ == "__main__":
    unittest.main()
