from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.training.execution import (
    TRAINING_SUCCESS_STATUS,
    TrainingExecutionError,
    validate_training_output_artifacts,
)


def _valid_run_provenance(run_dir: Path) -> dict[str, object]:
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
        "timestamp_utc": "2026-04-10T00:00:00Z",
    }


def test_successful_training_outputs_require_all_checkpoint_backed_artifacts(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint-1"
    checkpoint_dir.mkdir()
    (tmp_path / "run_provenance.json").write_text(
        json.dumps(_valid_run_provenance(tmp_path), indent=2) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "train_summary.json").write_text(
        json.dumps(
            {
                "status": TRAINING_SUCCESS_STATUS,
                "backend": "uniform",
                "task": "subject_personalization",
                "allocation_manifest": str(tmp_path / "uniform.json"),
                "train_steps": 1,
                "checkpoint_path": str(checkpoint_dir),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "global_step": 1,
                "train_loss_last": 1.0,
                "train_loss_mean": 1.0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "checkpoint_info.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_dir),
                "checkpoint_exists": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    payload = validate_training_output_artifacts(tmp_path)
    assert payload["train_summary"]["status"] == TRAINING_SUCCESS_STATUS


def test_planner_status_is_not_accepted_as_successful_training_output(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoint-1"
    checkpoint_dir.mkdir()
    (tmp_path / "run_provenance.json").write_text(
        json.dumps(_valid_run_provenance(tmp_path), indent=2) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "train_summary.json").write_text(
        json.dumps(
            {
                "status": "planning_only",
                "backend": "uniform",
                "task": "subject_personalization",
                "allocation_manifest": str(tmp_path / "uniform.json"),
                "train_steps": 1,
                "checkpoint_path": str(checkpoint_dir),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "global_step": 1,
                "train_loss_last": 1.0,
                "train_loss_mean": 1.0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TrainingExecutionError, match="required success artifacts|planning_only"):
        validate_training_output_artifacts(tmp_path)
