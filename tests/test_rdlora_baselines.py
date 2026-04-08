from __future__ import annotations

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

from rd_lora.baselines import (
    build_intlora_command,
    build_tlora_command,
    resolve_intlora_config,
    resolve_tlora_config,
    validate_intlora_launch_plan,
    validate_tlora_launch_plan,
    verify_registered_baseline,
)


TLORA_SCRIPT = REPO_ROOT / "scripts" / "run_tlora_baseline.py"
INTLORA_SCRIPT = REPO_ROOT / "scripts" / "run_intlora_baseline.py"


def _git(cmd: list[str], *, cwd: Path) -> str:
    completed = subprocess.run(
        ["git", *cmd],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_image(path: Path, *, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _init_checkout(path: Path, *, remote_url: str, files: dict[str, str]) -> str:
    path.mkdir(parents=True, exist_ok=True)
    _git(["init"], cwd=path)
    _git(["config", "user.email", "tests@example.com"], cwd=path)
    _git(["config", "user.name", "Tests"], cwd=path)
    for relative_path, content in files.items():
        file_path = path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
    _git(["add", "."], cwd=path)
    _git(["commit", "-m", "init"], cwd=path)
    _git(["remote", "add", "origin", remote_url], cwd=path)
    return _git(["rev-parse", "HEAD"], cwd=path)


class BaselineWrapperTests(unittest.TestCase):
    def test_tlora_config_and_command_validate_against_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "outputs").mkdir()
            (repo_root / "configs" / "accelerate").mkdir(parents=True)
            (repo_root / "configs" / "accelerate" / "single_gpu_fp16.yaml").write_text(
                "\n".join(
                    [
                        "compute_environment: LOCAL_MACHINE",
                        "distributed_type: 'NO'",
                        "gpu_ids: '0'",
                        "num_processes: 1",
                        "use_cpu: false",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            tlora_root = repo_root / "baselines" / "external" / "controlgenai_t-lora"
            _write_image(tlora_root / "dog_example" / "02.jpg", size=(32, 32), color=(50, 100, 150))
            commit = _init_checkout(
                tlora_root,
                remote_url="https://github.com/ControlGenAI/T-LoRA.git",
                files={
                    "train.py": "print('tlora')\n",
                    "inference.py": "print('infer')\n",
                    "README.md": "README\n",
                    "dog_example/.keep": "\n",
                },
            )

            registry = {
                "schema_version": 1,
                "baselines": [
                    {
                        "id": "t_lora",
                        "name": "T-LoRA",
                        "upstream_url": "https://github.com/ControlGenAI/T-LoRA.git",
                        "default_branch": "main",
                        "pinned_commit": commit,
                        "local_checkout_path": "baselines/external/controlgenai_t-lora",
                        "status": "available",
                        "critical_path": True,
                        "launch_notes": ["Official SDXL path keeps resolution 1024."],
                        "reason": None,
                    }
                ],
            }
            registry_path = repo_root / "baselines" / "registry.lock.json"
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            raw_config = {
                "paths": {
                    "repo_root": str(repo_root),
                    "registry_lock": str(registry_path),
                    "baseline_root": str(tlora_root),
                    "train_script": str(tlora_root / "train.py"),
                    "output_root": str(repo_root / "outputs" / "baselines" / "t_lora"),
                    "accelerate_config": str(repo_root / "configs" / "accelerate" / "single_gpu_fp16.yaml"),
                }
            }
            config = resolve_tlora_config(REPO_ROOT, raw_config)
            registration = verify_registered_baseline(config)
            self.assertTrue(registration["ok"])

            plan = build_tlora_command(config)
            validation = validate_tlora_launch_plan(plan, repo_root=repo_root)
            self.assertFalse(validation["accelerate_uses_cpu"])
            self.assertIn("--resolution", plan["command"])
            self.assertIn("--trainer_class", plan["command"])

    def test_intlora_config_and_command_validate_against_appendix_registry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            intlora_root = repo_root / "baselines" / "external" / "csguoh_intlora"
            _write_image(repo_root / "instance_data" / "sample.png", size=(32, 32), color=(20, 40, 60))
            commit = _init_checkout(
                intlora_root,
                remote_url="https://github.com/csguoh/IntLoRA.git",
                files={
                    "train_dreambooth_quant.py": "print('intlora')\n",
                    "evaluation.py": "print('eval')\n",
                    "get_results.py": "print('results')\n",
                    "README.md": "README\n",
                },
            )

            registry = {
                "schema_version": 1,
                "baselines": [
                    {
                        "id": "int_lora",
                        "name": "IntLoRA",
                        "upstream_url": "https://github.com/csguoh/IntLoRA.git",
                        "default_branch": "main",
                        "pinned_commit": commit,
                        "local_checkout_path": "baselines/external/csguoh_intlora",
                        "status": "appendix-only",
                        "critical_path": False,
                        "launch_notes": ["Official path is SD1.5 quantized DreamBooth at 512."],
                        "reason": "Appendix-only because the official substrate is SD1.5, not SDXL.",
                    }
                ],
            }
            registry_path = repo_root / "baselines" / "registry.lock.json"
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            raw_config = {
                "paths": {
                    "repo_root": str(repo_root),
                    "registry_lock": str(registry_path),
                    "baseline_root": str(intlora_root),
                    "train_script": str(intlora_root / "train_dreambooth_quant.py"),
                    "output_root": str(repo_root / "outputs" / "baselines" / "int_lora"),
                },
                "dataset": {
                    "instance_data_dir": str(repo_root / "instance_data"),
                },
            }
            config = resolve_intlora_config(REPO_ROOT, raw_config)
            registration = verify_registered_baseline(config)
            self.assertTrue(registration["ok"])

            plan = build_intlora_command(config)
            validation = validate_intlora_launch_plan(plan, repo_root=repo_root)
            self.assertTrue(validation["python_found"])
            self.assertIn("--intlora", plan["command"])
            self.assertIn("--resolution", plan["command"])

    def test_wrapper_scripts_write_dry_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            (repo_root / "outputs").mkdir()
            (repo_root / "configs" / "accelerate").mkdir(parents=True)
            (repo_root / "configs" / "accelerate" / "single_gpu_fp16.yaml").write_text(
                "use_cpu: false\n",
                encoding="utf-8",
            )

            tlora_root = repo_root / "baselines" / "external" / "controlgenai_t-lora"
            intlora_root = repo_root / "baselines" / "external" / "csguoh_intlora"
            _write_image(tlora_root / "dog_example" / "02.jpg", size=(32, 32), color=(10, 20, 30))
            _write_image(repo_root / "instance_data" / "sample.png", size=(32, 32), color=(30, 20, 10))
            tlora_commit = _init_checkout(
                tlora_root,
                remote_url="https://github.com/ControlGenAI/T-LoRA.git",
                files={
                    "train.py": "print('tlora')\n",
                    "inference.py": "print('infer')\n",
                    "README.md": "README\n",
                    "dog_example/.keep": "\n",
                },
            )
            intlora_commit = _init_checkout(
                intlora_root,
                remote_url="https://github.com/csguoh/IntLoRA.git",
                files={
                    "train_dreambooth_quant.py": "print('intlora')\n",
                    "evaluation.py": "print('eval')\n",
                    "get_results.py": "print('results')\n",
                    "README.md": "README\n",
                },
            )

            registry = {
                "schema_version": 1,
                "baselines": [
                    {
                        "id": "t_lora",
                        "name": "T-LoRA",
                        "upstream_url": "https://github.com/ControlGenAI/T-LoRA.git",
                        "default_branch": "main",
                        "pinned_commit": tlora_commit,
                        "local_checkout_path": "baselines/external/controlgenai_t-lora",
                        "status": "available",
                        "critical_path": True,
                        "launch_notes": ["Official SDXL path keeps resolution 1024."],
                        "reason": None,
                    },
                    {
                        "id": "int_lora",
                        "name": "IntLoRA",
                        "upstream_url": "https://github.com/csguoh/IntLoRA.git",
                        "default_branch": "main",
                        "pinned_commit": intlora_commit,
                        "local_checkout_path": "baselines/external/csguoh_intlora",
                        "status": "appendix-only",
                        "critical_path": False,
                        "launch_notes": ["Official path is SD1.5 quantized DreamBooth at 512."],
                        "reason": "Appendix-only because the official substrate is SD1.5, not SDXL.",
                    },
                ],
            }
            registry_path = repo_root / "baselines" / "registry.lock.json"
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            tlora_config_path = repo_root / "tlora.yaml"
            tlora_config_path.write_text(
                yaml.safe_dump(
                    {
                        "paths": {
                            "repo_root": str(repo_root),
                            "registry_lock": str(registry_path),
                            "baseline_root": str(tlora_root),
                            "train_script": str(tlora_root / "train.py"),
                            "output_root": str(repo_root / "outputs" / "baselines" / "t_lora"),
                            "accelerate_config": str(
                                repo_root / "configs" / "accelerate" / "single_gpu_fp16.yaml"
                            ),
                        }
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            intlora_config_path = repo_root / "intlora.yaml"
            intlora_config_path.write_text(
                yaml.safe_dump(
                    {
                        "paths": {
                            "repo_root": str(repo_root),
                            "registry_lock": str(registry_path),
                            "baseline_root": str(intlora_root),
                            "train_script": str(intlora_root / "train_dreambooth_quant.py"),
                            "output_root": str(repo_root / "outputs" / "baselines" / "int_lora"),
                        },
                        "dataset": {
                            "instance_data_dir": str(repo_root / "instance_data"),
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [sys.executable, str(TLORA_SCRIPT), "--config", str(tlora_config_path)],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [sys.executable, str(INTLORA_SCRIPT), "--config", str(intlora_config_path)],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue(
                (
                    repo_root
                    / "outputs"
                    / "baselines"
                    / "t_lora"
                    / "dog_example"
                    / "launch_summary.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    repo_root
                    / "outputs"
                    / "baselines"
                    / "int_lora"
                    / "sd15_appendix_smoke"
                    / "launch_summary.json"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
