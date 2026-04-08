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
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.substrate.diffusers_sdxl import (
    DEFAULT_IMAGE_SIZE,
    build_diffusers_sdxl_lora_command,
    default_vanilla_lora_config,
    load_rdlora_tasks,
    prepare_pilot_imagefolder,
    resolve_vanilla_lora_config,
    validate_launch_plan,
    validate_pilot_manifest,
    validate_pilot_task,
)


PREPARE_SCRIPT = REPO_ROOT / "scripts" / "prepare_rdlora_pilot_data.py"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_vanilla_lora.py"
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_sdxl_lora.py"


def _write_source_image(path: Path, *, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _write_config_pair(repo_root: Path) -> tuple[Path, Path]:
    source_root = repo_root / "pilot_sources"
    _write_source_image(source_root / "portrait_a.png", size=(1600, 1200), color=(220, 180, 150))
    _write_source_image(source_root / "portrait_b.png", size=(900, 1500), color=(120, 150, 210))
    accelerate_config_path = repo_root / "configs" / "accelerate" / "single_gpu_fp16.yaml"
    accelerate_config_path.parent.mkdir(parents=True, exist_ok=True)
    accelerate_config_path.write_text(
        "\n".join(
            [
                "compute_environment: LOCAL_MACHINE",
                "debug: false",
                "distributed_type: 'NO'",
                "enable_cpu_affinity: false",
                "gpu_ids: '0'",
                "machine_rank: 0",
                "main_training_function: main",
                "mixed_precision: fp16",
                "num_machines: 1",
                "num_processes: 1",
                "rdzv_backend: static",
                "same_network: true",
                "use_cpu: false",
                "",
            ]
        ),
        encoding="utf-8",
    )

    tasks_config = {
        "version": 1,
        "defaults": {
            "prepared_root": "outputs/rd_lora/pilot_data",
            "image_size": DEFAULT_IMAGE_SIZE,
            "image_column": "image",
            "caption_column": "text",
        },
        "tasks": {
            "smoke_portrait": {
                "concept_name": "unit_test_smoke_portrait",
                "validation_prompt": "A portrait of the same person with natural light.",
                "train_records": [
                    {
                        "image": str((source_root / "portrait_a.png").relative_to(repo_root)),
                        "caption": "A portrait photo of a person under soft studio light.",
                    },
                    {
                        "image": str((source_root / "portrait_b.png").relative_to(repo_root)),
                        "caption": "A close-up portrait of the same person in natural light.",
                    },
                ],
            }
        },
    }
    tasks_path = repo_root / "rdlora_tasks_test.yaml"
    with tasks_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(tasks_config, handle, sort_keys=False)

    raw_config = default_vanilla_lora_config()
    raw_config["paths"]["repo_root"] = str(repo_root)
    raw_config["paths"]["tasks_config"] = str(tasks_path)
    raw_config["accelerate"]["launch"]["config_file"] = str(accelerate_config_path.relative_to(repo_root))
    raw_config["paths"]["official_script"] = str(
        REPO_ROOT
        / "baselines"
        / "external"
        / "huggingface_diffusers"
        / "examples"
        / "text_to_image"
        / "train_text_to_image_lora_sdxl.py"
    )
    raw_config["smoke"]["gradient_checkpointing"] = True
    raw_config["smoke"]["mixed_precision"] = "fp16"
    config_path = repo_root / "rdlora_vanilla_test.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_config, handle, sort_keys=False)
    return config_path, tasks_path


class DiffusersSdxlHarnessTests(unittest.TestCase):
    def test_prepare_pilot_imagefolder_is_deterministic_and_writes_manifest_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            config_path, tasks_path = _write_config_pair(repo_root)
            config = resolve_vanilla_lora_config(REPO_ROOT, yaml.safe_load(config_path.read_text()))
            tasks_bundle = load_rdlora_tasks(tasks_path)

            summary = prepare_pilot_imagefolder(config, tasks_bundle, "smoke_portrait")
            manifest_path = Path(summary["manifest_path"])
            first_manifest = manifest_path.read_text(encoding="utf-8")

            second_summary = prepare_pilot_imagefolder(config, tasks_bundle, "smoke_portrait")
            second_manifest = manifest_path.read_text(encoding="utf-8")
            self.assertEqual(summary["image_count"], 2)
            self.assertEqual(second_summary["image_count"], 2)
            self.assertEqual(first_manifest, second_manifest)

            manifest_summary = validate_pilot_manifest(
                manifest_path,
                repo_root=repo_root,
                verify_files=True,
            )
            self.assertEqual(manifest_summary["total"], 2)

            with manifest_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)

            metadata_lines = Path(summary["metadata_path"]).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(metadata_lines), 2)
            first_payload = json.loads(metadata_lines[0])
            self.assertIn("file_name", first_payload)
            self.assertIn("text", first_payload)

            for row in rows:
                prepared_path = repo_root / row["prepared_path"]
                with Image.open(prepared_path) as image:
                    self.assertEqual(image.size, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

    def test_validate_pilot_task_rejects_missing_caption(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            source_path = repo_root / "portrait.png"
            _write_source_image(source_path, size=(512, 512), color=(100, 120, 140))
            tasks_bundle = {
                "path": str(repo_root / "bad_tasks.yaml"),
                "defaults": {
                    "image_size": DEFAULT_IMAGE_SIZE,
                    "image_column": "image",
                    "caption_column": "text",
                },
                "tasks": {
                    "broken_task": {
                        "concept_name": "broken",
                        "validation_prompt": "A broken prompt.",
                        "train_records": [
                            {
                                "image": str(source_path.relative_to(repo_root)),
                                "caption": "",
                            }
                        ],
                    }
                },
            }
            with self.assertRaisesRegex(Exception, "caption"):
                validate_pilot_task(repo_root, tasks_bundle, "broken_task")

    def test_build_diffusers_command_wraps_official_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            config_path, tasks_path = _write_config_pair(repo_root)
            config = resolve_vanilla_lora_config(REPO_ROOT, yaml.safe_load(config_path.read_text()))
            tasks_bundle = load_rdlora_tasks(tasks_path)
            prepare_pilot_imagefolder(config, tasks_bundle, "smoke_portrait")

            plan = build_diffusers_sdxl_lora_command(config, task_id="smoke_portrait", smoke=False)
            validation = validate_launch_plan(plan, repo_root=repo_root)
            self.assertTrue(validation["wrapped_official_script"])
            self.assertEqual(plan["command"][:2], ["accelerate", "launch"])
            self.assertIn(
                "baselines/external/huggingface_diffusers/examples/text_to_image/train_text_to_image_lora_sdxl.py",
                plan["manual_command"],
            )
            self.assertIn("--config_file", plan["command"])
            self.assertIn("--train_data_dir", plan["command"])
            self.assertIn("--pretrained_model_name_or_path", plan["command"])
            self.assertIn("--resolution", plan["command"])
            self.assertNotIn(str(TRAIN_SCRIPT), plan["command"])

    def test_prepare_and_train_scripts_dry_run_validate_launch_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            config_path, _ = _write_config_pair(repo_root)

            subprocess.run(
                [sys.executable, str(PREPARE_SCRIPT), "--config", str(config_path), "--task-id", "smoke_portrait"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            completed = subprocess.run(
                [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path), "--task-id", "smoke_portrait"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            output_dir = repo_root / "outputs" / "rd_lora" / "vanilla" / "smoke_portrait" / "vanilla_lora"
            self.assertTrue((output_dir / "resolved_config.yaml").is_file())
            self.assertTrue((output_dir / "launch_summary.json").is_file())
            self.assertTrue((output_dir / "launch_command.sh").is_file())
            self.assertIn("manual_gpu_command=", completed.stdout)
            self.assertIn("wrapped_official_script=1", completed.stdout)

    def test_smoke_command_uses_smoke_overrides_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            config_path, _ = _write_config_pair(repo_root)

            config = resolve_vanilla_lora_config(REPO_ROOT, yaml.safe_load(config_path.read_text()))
            tasks_bundle = load_rdlora_tasks(config["paths"]["tasks_config"])
            prepare_pilot_imagefolder(config, tasks_bundle, "smoke_portrait")
            plan = build_diffusers_sdxl_lora_command(config, task_id="smoke_portrait", smoke=True)
            self.assertIn("--config_file", plan["command"])
            self.assertIn("--gradient_checkpointing", plan["command"])
            self.assertIn("fp16", plan["command"])

            completed = subprocess.run(
                [sys.executable, str(SMOKE_SCRIPT), "--config", str(config_path), "--task-id", "smoke_portrait"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            output_dir = repo_root / "outputs" / "rd_lora" / "smoke" / "smoke_portrait" / "smoke"
            self.assertTrue((output_dir / "resolved_config.yaml").is_file())
            self.assertTrue((output_dir / "smoke_summary.json").is_file())
            self.assertTrue((output_dir / "manual_gpu_command.sh").is_file())
            shell_command = (output_dir / "manual_gpu_command.sh").read_text(encoding="utf-8")
            self.assertIn("--config_file", shell_command)
            self.assertIn("--gradient_checkpointing", shell_command)
            self.assertIn("manual_gpu_command=", completed.stdout)
            self.assertIn("wrapped_official_script=1", completed.stdout)


if __name__ == "__main__":
    unittest.main()
