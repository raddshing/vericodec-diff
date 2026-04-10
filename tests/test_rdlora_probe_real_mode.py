from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.probe import ProbeValidationError, build_run_request, default_probe_config, dispatch_probe


RUN_SCRIPT = REPO_ROOT / "scripts" / "run_rdlora_probe.py"


def test_real_gpu_mode_hard_fails_when_cuda_unavailable(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--config",
            "configs/rdlora_probe.yaml",
            "--task",
            "subject_personalization",
            "--run_mode",
            "real_gpu",
            "--output_dir",
            str(tmp_path / "real_gpu_fail"),
            "--max_cells",
            "1",
            "--candidate_ranks",
            "0",
            "--probe_inner_steps",
            "1",
            "--max_train_batches",
            "1",
            "--max_val_batches",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "torch.cuda.is_available()" in completed.stderr


def test_cpu_mock_mode_is_only_allowed_when_explicitly_requested(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--config",
            "configs/rdlora_probe.yaml",
            "--task",
            "subject_personalization",
            "--output_dir",
            str(tmp_path / "missing_run_mode"),
            "--max_cells",
            "1",
            "--candidate_ranks",
            "0",
            "--probe_inner_steps",
            "1",
            "--max_train_batches",
            "1",
            "--max_val_batches",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "--run_mode" in completed.stderr


def test_real_gpu_dispatch_never_falls_back_to_cpu_mock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = build_run_request(
        config=default_probe_config(),
        task="subject_personalization",
        run_mode="real_gpu",
        output_dir=tmp_path,
        max_cells=1,
        candidate_ranks=(0,),
        probe_inner_steps=1,
        max_train_batches=1,
        max_val_batches=1,
    )

    def _fail_cpu_mock(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("cpu_mock runner must not be reachable from real_gpu mode")

    monkeypatch.setattr("rd_lora.probe.run_cpu_mock_probe", _fail_cpu_mock)

    with pytest.raises(ProbeValidationError, match="torch.cuda.is_available"):
        dispatch_probe(config)


def test_real_gpu_probe_source_has_no_hash_utility_path() -> None:
    source = (SRC_ROOT / "rd_lora" / "probe.py").read_text(encoding="utf-8")
    assert "hashlib" not in source
    assert "_stable_" not in source
