from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "assert_real_run.py"


def _valid_provenance(run_dir: Path) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "run_mode": "real_gpu",
        "used_gpu": True,
        "python": "3.11.0",
        "torch": "2.5.0",
        "torch_cuda_is_available": True,
        "torch_cuda_version": "12.1",
        "torch_device_count": 1,
        "gpu_names": ["Fake GPU 0"],
        "diffusers": "0.0.test",
        "diffusers_file": str(run_dir / "fake_diffusers.py"),
        "accelerate_config_file": str(run_dir / "single_gpu_fp16.yaml"),
        "git_commit": "abc123",
        "backend": "uniform",
        "task": "subject_personalization",
        "allocation_manifest": str(run_dir / "uniform.json"),
        "peak_vram_mib": 512.0,
        "timestamp_utc": "2026-04-09T00:00:00Z",
        "source_run_dirs": [str(run_dir / "upstream_real")],
    }


def test_assert_real_run_accepts_valid_real_gpu_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "uniform_run"
    run_dir.mkdir()
    (run_dir / "train_summary.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_provenance.json").write_text(
        json.dumps(_valid_provenance(run_dir), indent=2) + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run_dir",
            str(run_dir),
            "--require-gpu",
            "--forbid-mock",
            "--require-file",
            "train_summary.json",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_assert_real_run_rejects_mock_source_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "uniform_run"
    run_dir.mkdir()
    provenance = _valid_provenance(run_dir)
    provenance["source_run_dirs"] = [str(tmp_path / "stageb_cpu_smoke")]
    (run_dir / "run_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run_dir",
            str(run_dir),
            "--forbid-mock",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
