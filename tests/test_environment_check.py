from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from importlib import metadata
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vericodec_diff.environment_check import (
    build_environment_report,
    collect_expected_packages,
    render_environment_report_markdown,
    resolve_runtime_config,
)


class _FakeCuda:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 2

    def get_device_name(self, index: int) -> str:
        return ["GPU-A", "GPU-B"][index]


class _FakeTorchVersion:
    cuda = "12.1"


class _FakeTorch:
    __version__ = "2.4.1"
    cuda = _FakeCuda()
    version = _FakeTorchVersion()


class EnvironmentCheckTests(unittest.TestCase):
    def test_collect_expected_packages_applies_aliases_and_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            env_dir = repo_root / "environment"
            env_dir.mkdir(parents=True, exist_ok=True)

            env_file = env_dir / "test-env.yaml"
            env_file.write_text(
                yaml.safe_dump(
                    {
                        "dependencies": [
                            "python=3.10",
                            "pytorch=2.4.1",
                            "numpy=1.26.4",
                            "pillow",
                            {"pip": ["diffusers>=0.32.0,<0.36", "huggingface_hub>=0.24"]},
                        ]
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            config = resolve_runtime_config(
                repo_root,
                {
                    "paths": {"repo_root": ".", "output_dir": "logs/tests", "disk_checks": []},
                    "packages": {
                        "environment_files": ["environment/test-env.yaml"],
                        "exclude_names": ["python"],
                        "distribution_candidates": {"pytorch": ["torch"], "pillow": ["Pillow"]},
                    },
                    "checkpoints": [],
                },
            )

            packages = collect_expected_packages(config)
            self.assertEqual([item["name"] for item in packages], ["diffusers", "huggingface_hub", "numpy", "pillow", "pytorch"])
            self.assertEqual(packages[-1]["candidates"], ["torch"])

    def test_build_environment_report_tracks_missing_packages_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            existing_dir = repo_root / "existing"
            existing_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = repo_root / "checkpoints" / "model.pt"

            env_dir = repo_root / "environment"
            env_dir.mkdir(parents=True, exist_ok=True)
            env_file = env_dir / "test-env.yaml"
            env_file.write_text(
                yaml.safe_dump(
                    {"dependencies": ["numpy", "pytorch", {"pip": ["diffusers"]}]},
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            config = resolve_runtime_config(
                repo_root,
                {
                    "paths": {
                        "repo_root": ".",
                        "output_dir": "logs/tests",
                        "disk_checks": [
                            {"name": "repo_root", "path": ".", "must_exist": True},
                            {"name": "optional_data", "path": "missing-data", "must_exist": False},
                        ],
                    },
                    "packages": {
                        "environment_files": ["environment/test-env.yaml"],
                        "exclude_names": [],
                        "distribution_candidates": {"pytorch": ["torch"]},
                    },
                    "checkpoints": [{"name": "model", "path": "checkpoints/model.pt", "type": "file"}],
                },
            )

            versions = {"numpy": "1.26.4", "torch": "2.4.1"}

            def version_getter(name: str) -> str:
                if name not in versions:
                    raise metadata.PackageNotFoundError(name)
                return versions[name]

            def fake_disk_usage(path: str | Path) -> shutil._ntuple_diskusage:
                return shutil._ntuple_diskusage(total=100, used=40, free=60)

            report = build_environment_report(
                config,
                torch_module=_FakeTorch(),
                version_getter=version_getter,
                disk_usage_func=fake_disk_usage,
            )

            self.assertEqual(report["summary"]["status"], "issues_found")
            self.assertIn("1 required Python package(s) are missing.", report["summary"]["issues"])
            self.assertIn("1 required checkpoint(s) are missing.", report["summary"]["issues"])
            self.assertEqual(report["torch"]["gpu_names"], ["GPU-A", "GPU-B"])
            self.assertEqual(report["disk"]["items"][1]["status"], "skipped_missing")
            self.assertEqual(report["packages"]["summary"]["missing_names"], ["diffusers"])
            self.assertEqual(report["checkpoints"]["summary"]["missing_required"], 1)

            markdown = render_environment_report_markdown(report)
            self.assertIn("# Environment Report", markdown)
            self.assertIn("| diffusers | `no` | `n/a` |", markdown)
            self.assertIn("| model |", markdown)

    def test_report_payload_is_json_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            env_dir = repo_root / "environment"
            env_dir.mkdir(parents=True, exist_ok=True)
            env_file = env_dir / "test-env.yaml"
            env_file.write_text(yaml.safe_dump({"dependencies": ["numpy"]}, sort_keys=False), encoding="utf-8")

            config = resolve_runtime_config(
                repo_root,
                {
                    "paths": {"repo_root": ".", "output_dir": "logs/tests", "disk_checks": []},
                    "packages": {"environment_files": ["environment/test-env.yaml"], "exclude_names": []},
                    "checkpoints": [],
                },
            )

            def version_getter(name: str) -> str:
                if name != "numpy":
                    raise metadata.PackageNotFoundError(name)
                return "1.26.4"

            report = build_environment_report(
                config,
                torch_module=_FakeTorch(),
                version_getter=version_getter,
                disk_usage_func=lambda path: shutil._ntuple_diskusage(total=10, used=1, free=9),
            )

            payload = json.dumps(report, sort_keys=True)
            self.assertIn('"schema_version": 1', payload)


if __name__ == "__main__":
    unittest.main()
