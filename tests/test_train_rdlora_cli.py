from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_rdlora.py"


def test_train_rdlora_help_preserves_exact_cli_surface() -> None:
    completed = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    help_text = completed.stdout
    assert "--config" in help_text
    assert "--task {subject_personalization,style_domain}" in help_text
    assert "--backend {uniform,layer_only,timestep_only,proposed}" in help_text
    assert "--allocation" in help_text
    assert "--run_mode {real_gpu}" in help_text
    assert "--output_dir" in help_text
