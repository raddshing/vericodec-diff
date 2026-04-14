from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.eval.score import backend_score, style_score, subject_score


AGGREGATE_TASK = "__aggregate__"


def _extract_metric_like_gate(
    rows: list[dict[str, object]],
    *,
    backend: str,
    task: str,
    metric_name: str,
) -> float | None:
    for row in rows:
        if (
            str(row.get("backend")) == backend
            and str(row.get("task")) == task
            and str(row.get("metric_name")) == metric_name
        ):
            value = row.get("metric_value")
            if value is not None and value != "":
                return float(value)
    return None


def test_subject_metric_json_contains_score() -> None:
    metrics = {
        "dino_ref_max_mean": 0.80,
        "clip_i_ref_max_mean": 0.70,
        "clip_t_mean": 0.90,
        "score": subject_score(0.80, 0.70, 0.90),
        "num_images": 3,
    }

    assert "score" in metrics
    assert isinstance(metrics["score"], float)
    assert 0.0 <= metrics["score"] <= 1.0


def test_style_metric_json_contains_score() -> None:
    metrics = {
        "ocr_exact_match": 0.80,
        "ocr_cer_mean": 0.10,
        "ocr_text_score": 0.87,
        "clip_t_mean": 0.75,
        "score": style_score(0.80, 0.10, 0.75),
        "num_images": 3,
    }

    assert "score" in metrics
    assert isinstance(metrics["score"], float)
    assert 0.0 <= metrics["score"] <= 1.0


def test_long_form_rows_include_score_for_each_backend_and_aggregate() -> None:
    per_task_scores = {
        "uniform": {"subject_personalization": 0.40, "style_domain": 0.60},
        "layer_only": {"subject_personalization": 0.45, "style_domain": 0.55},
        "timestep_only": {"subject_personalization": 0.42, "style_domain": 0.58},
        "proposed": {"subject_personalization": 0.50, "style_domain": 0.70},
    }
    rows: list[dict[str, object]] = []
    for backend, task_scores in per_task_scores.items():
        for task, score in task_scores.items():
            rows.append(
                {
                    "task": task,
                    "backend": backend,
                    "metric_name": "score",
                    "metric_value": score,
                }
            )
        rows.append(
            {
                "task": AGGREGATE_TASK,
                "backend": backend,
                "metric_name": "score",
                "metric_value": backend_score(task_scores),
            }
        )

    for backend in per_task_scores:
        assert any(
            row["backend"] == backend and row["metric_name"] == "score"
            for row in rows
        )

    assert any(
        row["task"] == AGGREGATE_TASK and row["metric_name"] == "score"
        for row in rows
    )


def test_long_form_baseline_metrics_json_does_not_require_wide_score_column(tmp_path: Path) -> None:
    rows = [
        {
            "task": "subject_personalization",
            "backend": "uniform",
            "metric_name": "clip_t_mean",
            "metric_value": 0.72,
        },
        {
            "task": AGGREGATE_TASK,
            "backend": "uniform",
            "metric_name": "score",
            "metric_value": 0.81,
        },
    ]
    baseline_metrics_path = tmp_path / "baseline_metrics.json"
    baseline_metrics_path.write_text(
        json.dumps({"schema_version": "1.0", "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    payload = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    loaded_rows = payload["rows"]

    assert all("score" not in row for row in loaded_rows)
    assert _extract_metric_like_gate(
        loaded_rows,
        backend="uniform",
        task=AGGREGATE_TASK,
        metric_name="score",
    ) == 0.81
