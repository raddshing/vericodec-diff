from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_rdlora.py"


@pytest.mark.parametrize("backend_name", ["uniform", "proposed"])
def test_train_rdlora_cli_writes_planning_outputs(tmp_path: Path, backend_name: str) -> None:
    output_dir = tmp_path / backend_name
    allocation_path = REPO_ROOT / "outputs" / "rd_lora" / "allocation" / "gate_real" / f"{backend_name}.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--config",
            "configs/rdlora_vanilla.yaml",
            "--task",
            "subject_personalization",
            "--backend",
            backend_name,
            "--allocation",
            str(allocation_path),
            "--run_mode",
            "real_gpu",
            "--output_dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "contract_binding " in completed.stdout
    assert "backend_plan " in completed.stdout
    assert "planning_only=1" in completed.stdout
    assert (output_dir / "resolved_config.yaml").is_file()
    assert (output_dir / "adapter_plan.json").is_file()
    assert (output_dir / "train_summary.json").is_file()
    assert not (output_dir / "train.log").exists()

    summary = json.loads((output_dir / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "planning_only"
    assert summary["backend"] == backend_name
    assert summary["routing_mode"] in {"static", "timestep_band"}
