from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "write_rdlora_gate_memo.py"


def _write_stage_inputs(base_dir: Path, *, run_dir_name: str = "uniform_run", run_mode: str = "real_gpu") -> tuple[Path, Path, Path, Path]:
    probe_dir = base_dir / "probe"
    surrogate_dir = base_dir / "surrogate"
    allocation_dir = base_dir / "allocation"
    evaluation_dir = base_dir / "evaluation"
    probe_dir.mkdir()
    surrogate_dir.mkdir()
    allocation_dir.mkdir()
    evaluation_dir.mkdir()
    (probe_dir / "probe_summary.json").write_text(
        json.dumps({"task": "subject_personalization", "cell_count": 24, "candidate_ranks": [0, 2, 4, 8, 16], "row_count": 120, "wall_time_seconds": 1.0}),
        encoding="utf-8",
    )
    (probe_dir / "cell_utility.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"cell_id": "c0", "utility_score": 0.9},
                    {"cell_id": "c1", "utility_score": 0.1},
                ]
            }
        ),
        encoding="utf-8",
    )
    (surrogate_dir / "surrogate_summary.json").write_text(
        json.dumps({"held_out_spearman_rho": 0.7, "top_5_precision": 0.8, "train_rows": 96, "val_rows": 24, "wall_time_seconds": 1.0}),
        encoding="utf-8",
    )
    (allocation_dir / "allocation_summary.json").write_text(
        json.dumps({"rank_budget_total": 96, "manifest_files": {}, "backends": ["uniform"], "wall_time_seconds": 1.0}),
        encoding="utf-8",
    )
    run_dir = base_dir / run_dir_name
    run_dir.mkdir()
    (run_dir / "run_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_mode": run_mode,
                "used_gpu": True,
                "python": "3.11.0",
                "torch": "2.5.0",
                "torch_cuda_is_available": True,
                "torch_cuda_version": "12.1",
                "torch_device_count": 1,
                "gpu_names": ["Fake GPU 0"],
                "diffusers": "0.0.test",
                "diffusers_file": str(run_dir / "diffusers.py"),
                "accelerate_config_file": str(run_dir / "cfg.yaml"),
                "git_commit": "abc123",
                "backend": "uniform",
                "task": "subject_personalization",
                "allocation_manifest": str(run_dir / "uniform.json"),
                "peak_vram_mib": 512.0,
                "timestamp_utc": "2026-04-09T00:00:00Z",
                "source_run_dirs": [str(base_dir / "real_parent")],
            }
        ),
        encoding="utf-8",
    )
    (evaluation_dir / "evaluation_summary.json").write_text(
        json.dumps(
            {
                "tasks": ["subject_personalization"],
                "backends": ["uniform"],
                "metric_names": ["score", "wall_time_seconds"],
                "evaluated_run_dirs": [str(run_dir)],
            }
        ),
        encoding="utf-8",
    )
    (evaluation_dir / "baseline_metrics.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"backend": "uniform", "task": "subject_personalization", "run_dir": str(run_dir), "score": 1.0, "wall_time_seconds": 2.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    return probe_dir, surrogate_dir, allocation_dir, evaluation_dir


def test_gate_status_is_invalid_when_required_baselines_are_missing(tmp_path: Path) -> None:
    probe_dir, surrogate_dir, allocation_dir, evaluation_dir = _write_stage_inputs(tmp_path)
    output_dir = tmp_path / "gate"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--probe_dir",
            str(probe_dir),
            "--surrogate_dir",
            str(surrogate_dir),
            "--allocation_dir",
            str(allocation_dir),
            "--evaluation_dir",
            str(evaluation_dir),
            "--output_dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output_dir / "gate_memo.json").read_text(encoding="utf-8"))
    assert payload["gate_status"] == "INVALID"
    assert any("Missing required baselines" in reason for reason in payload["invalid_reasons"])


def test_gate_status_is_invalid_for_non_real_run_inputs(tmp_path: Path) -> None:
    probe_dir, surrogate_dir, allocation_dir, evaluation_dir = _write_stage_inputs(
        tmp_path,
        run_dir_name="real_run",
        run_mode="mock",
    )
    rows = json.loads((evaluation_dir / "baseline_metrics.json").read_text(encoding="utf-8"))["rows"]
    for backend in ("layer_only", "timestep_only", "proposed"):
        rows.append({"backend": backend, "task": "subject_personalization", "run_dir": rows[0]["run_dir"], "score": 1.0, "wall_time_seconds": 2.0})
    (evaluation_dir / "baseline_metrics.json").write_text(json.dumps({"rows": rows}), encoding="utf-8")
    (evaluation_dir / "evaluation_summary.json").write_text(
        json.dumps(
            {
                "tasks": ["subject_personalization"],
                "backends": ["uniform", "layer_only", "timestep_only", "proposed"],
                "metric_names": ["score", "wall_time_seconds"],
                "evaluated_run_dirs": [rows[0]["run_dir"]],
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "gate"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--probe_dir",
            str(probe_dir),
            "--surrogate_dir",
            str(surrogate_dir),
            "--allocation_dir",
            str(allocation_dir),
            "--evaluation_dir",
            str(evaluation_dir),
            "--output_dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output_dir / "gate_memo.json").read_text(encoding="utf-8"))
    assert payload["gate_status"] == "INVALID"
    assert any("Non-real input run" in reason for reason in payload["invalid_reasons"])
