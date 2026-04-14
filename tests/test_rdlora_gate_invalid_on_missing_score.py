from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "write_rdlora_gate_memo.py"
AGGREGATE_TASK = "__aggregate__"
BACKENDS = ["uniform", "layer_only", "timestep_only", "proposed"]
TASKS = ["subject_personalization", "style_domain"]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_run_provenance(run_dir: Path, *, backend: str) -> None:
    _write_json(
        run_dir / "run_provenance.json",
        {
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
            "diffusers_file": str(run_dir / "diffusers.py"),
            "accelerate_config_file": str(run_dir / "accelerate.yaml"),
            "git_commit": "abc123",
            "backend": backend,
            "task": "subject_personalization",
            "allocation_manifest": str(run_dir / f"{backend}.json"),
            "peak_vram_mib": 9500.0,
            "timestamp_utc": "2026-04-14T00:00:00Z",
        },
    )


def _write_probe_surrogate_allocation_dirs(
    base_dir: Path,
    *,
    surrogate_summary: dict[str, object] | None = None,
) -> tuple[Path, Path, Path]:
    probe_dir = base_dir / "probe"
    surrogate_dir = base_dir / "surrogate"
    allocation_dir = base_dir / "allocation"
    _write_json(probe_dir / "probe_summary.json", {"wall_time_seconds": 10.0})
    _write_json(
        probe_dir / "cell_utility.json",
        {"rows": [{"cell_id": "c1", "utility_score": 0.5}]},
    )
    _write_json(
        surrogate_dir / "surrogate_summary.json",
        surrogate_summary
        or {
            "held_out_spearman_rho": 0.0,
            "top_5_precision": 1.0,
            "wall_time_seconds": 5.0,
        },
    )
    _write_json(allocation_dir / "allocation_summary.json", {"wall_time_seconds": 2.0})
    return probe_dir, surrogate_dir, allocation_dir


def _write_evaluation_dir(
    base_dir: Path,
    *,
    metric_names: list[str],
    rows: list[dict[str, object]],
) -> Path:
    evaluation_dir = base_dir / "evaluation"
    run_dirs: list[str] = []
    for backend in BACKENDS:
        run_dir = base_dir / f"{backend}_run"
        _write_run_provenance(run_dir, backend=backend)
        run_dirs.append(str(run_dir))

    _write_json(
        evaluation_dir / "evaluation_summary.json",
        {
            "schema_version": "1.0",
            "tasks": TASKS,
            "backends": BACKENDS,
            "metric_names": metric_names,
            "evaluated_run_dirs": run_dirs,
            "primary_metric": "score",
        },
    )
    _write_json(
        evaluation_dir / "baseline_metrics.json",
        {
            "schema_version": "1.0",
            "rows": rows,
        },
    )
    return evaluation_dir


def _run_gate(
    *,
    probe_dir: Path,
    surrogate_dir: Path,
    allocation_dir: Path,
    evaluation_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
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
    return json.loads((output_dir / "gate_memo.json").read_text(encoding="utf-8"))


def test_gate_is_invalid_when_score_missing_from_long_form_rows(tmp_path: Path) -> None:
    probe_dir, surrogate_dir, allocation_dir = _write_probe_surrogate_allocation_dirs(tmp_path)
    rows = [
        {
            "task": AGGREGATE_TASK,
            "backend": backend,
            "metric_name": "train_loss_last",
            "metric_value": 1.23,
        }
        for backend in BACKENDS
    ]
    evaluation_dir = _write_evaluation_dir(
        tmp_path,
        metric_names=["train_loss_last"],
        rows=rows,
    )

    payload = _run_gate(
        probe_dir=probe_dir,
        surrogate_dir=surrogate_dir,
        allocation_dir=allocation_dir,
        evaluation_dir=evaluation_dir,
        output_dir=tmp_path / "output_missing_score",
    )

    assert payload["gate_status"] == "INVALID"
    invalid_reasons = [str(reason) for reason in payload["invalid_reasons"]]
    assert any("score" in reason or "primary_metric" in reason for reason in invalid_reasons)


def test_gate_is_invalid_when_uniform_wall_time_rows_are_missing(tmp_path: Path) -> None:
    probe_dir, surrogate_dir, allocation_dir = _write_probe_surrogate_allocation_dirs(tmp_path)
    rows = [
        {
            "task": AGGREGATE_TASK,
            "backend": backend,
            "metric_name": "score",
            "metric_value": 0.20,
        }
        for backend in BACKENDS
    ]
    evaluation_dir = _write_evaluation_dir(
        tmp_path,
        metric_names=["score"],
        rows=rows,
    )

    payload = _run_gate(
        probe_dir=probe_dir,
        surrogate_dir=surrogate_dir,
        allocation_dir=allocation_dir,
        evaluation_dir=evaluation_dir,
        output_dir=tmp_path / "output_missing_wall_time",
    )

    assert payload["gate_status"] == "INVALID"
    invalid_reasons = [str(reason) for reason in payload["invalid_reasons"]]
    assert any("wall_time_sec" in reason for reason in invalid_reasons)


def test_gate_is_valid_no_go_when_metrics_exist_but_thresholds_fail(tmp_path: Path) -> None:
    probe_dir, surrogate_dir, allocation_dir = _write_probe_surrogate_allocation_dirs(
        tmp_path,
        surrogate_summary={
            "held_out_spearman_rho": 0.0,
            "top_5_precision": 1.0,
            "wall_time_seconds": 5.0,
        },
    )
    rows = [
        {
            "task": AGGREGATE_TASK,
            "backend": "uniform",
            "metric_name": "score",
            "metric_value": 0.10,
        },
        {
            "task": AGGREGATE_TASK,
            "backend": "layer_only",
            "metric_name": "score",
            "metric_value": 0.10,
        },
        {
            "task": AGGREGATE_TASK,
            "backend": "timestep_only",
            "metric_name": "score",
            "metric_value": 0.10,
        },
        {
            "task": AGGREGATE_TASK,
            "backend": "proposed",
            "metric_name": "score",
            "metric_value": 0.10,
        },
        {
            "task": "subject_personalization",
            "backend": "uniform",
            "metric_name": "wall_time_sec",
            "metric_value": 10.0,
        },
        {
            "task": "style_domain",
            "backend": "uniform",
            "metric_name": "wall_time_sec",
            "metric_value": 10.0,
        },
    ]
    evaluation_dir = _write_evaluation_dir(
        tmp_path,
        metric_names=["score", "wall_time_sec"],
        rows=rows,
    )

    payload = _run_gate(
        probe_dir=probe_dir,
        surrogate_dir=surrogate_dir,
        allocation_dir=allocation_dir,
        evaluation_dir=evaluation_dir,
        output_dir=tmp_path / "output_valid_no_go",
    )

    assert payload["gate_status"] == "VALID_NO_GO"
