from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.runtime.provenance import (
    FORBIDDEN_SOURCE_TOKENS,
    assert_real_gpu_provenance,
    iter_recorded_source_paths,
    load_run_provenance,
    path_contains_forbidden_token,
)
from rd_lora.substrate.diffusers_sdxl import deep_update, load_yaml_mapping, save_json
from vericodec_diff.config import OmegaConf


DEFAULT_CONFIG_PATH = "configs/rdlora_gate.yaml"
AGGREGATE_TASK = "__aggregate__"
SCORE_METRIC_NAME = "score"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write the RD-LoRA gate memo with ternary status: VALID_GO, VALID_NO_GO, or INVALID."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config path.")
    parser.add_argument("--probe_dir", required=True, help="Probe output directory.")
    parser.add_argument("--surrogate_dir", required=True, help="Surrogate output directory.")
    parser.add_argument("--allocation_dir", required=True, help="Allocation output directory.")
    parser.add_argument("--evaluation_dir", required=True, help="Evaluation output directory.")
    parser.add_argument("--output_dir", required=True, help="Output directory for gate_memo.md/json.")
    return parser.parse_args()


def _default_config() -> dict[str, Any]:
    return {
        "thresholds": {
            "top_20_mass_ratio": 0.60,
            "held_out_spearman_rho": 0.50,
            "top_5_precision": 0.60,
            "proposed_relative_vs_uniform": 0.10,
            "probe_allocation_overhead_ratio": 1.5,
        },
        "policies": {
            "required_backends": ["uniform", "layer_only", "timestep_only", "proposed"],
            "primary_metric": "score",
            "forbidden_path_tokens": list(FORBIDDEN_SOURCE_TOKENS),
        },
    }


def _load_config(path_value: str) -> dict[str, Any]:
    config_path = Path(path_value).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    raw = load_yaml_mapping(config_path) if config_path.is_file() else {}
    return deep_update(_default_config(), raw)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _top_20_mass_ratio(cell_rows: Sequence[Mapping[str, Any]]) -> float:
    best_by_cell: dict[str, float] = {}
    for row in cell_rows:
        best_by_cell[str(row["cell_id"])] = max(
            best_by_cell.get(str(row["cell_id"]), 0.0),
            float(row["utility_score"]),
        )
    if not best_by_cell:
        return 0.0
    values = sorted(best_by_cell.values(), reverse=True)
    top_count = max(1, math.ceil(len(values) * 0.20))
    total = sum(values)
    if total <= 0.0:
        return 0.0
    return round(sum(values[:top_count]) / total, 6)


def _load_long_form_rows(evaluation_dir: Path) -> list[dict[str, Any]]:
    payload = _read_json(evaluation_dir / "baseline_metrics.json")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("baseline_metrics.json must contain rows")
    return [dict(row) for row in rows]


def _extract_metric(
    rows: list[dict[str, Any]],
    *,
    backend: str,
    task: str,
    metric_name: str,
) -> float | None:
    """Find a specific metric from long-form rows. Return None if not found."""
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


def _status_and_summary(
    *,
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    invalid_reasons: list[str],
) -> tuple[str, str]:
    if invalid_reasons or any(value is None for value in metrics.values()):
        return "INVALID", "Gate input set is invalid."

    checks = [
        ("top_20_mass_ratio", float(metrics["top_20_mass_ratio"]) >= float(thresholds["top_20_mass_ratio"])),
        (
            "held_out_spearman_rho",
            float(metrics["held_out_spearman_rho"]) >= float(thresholds["held_out_spearman_rho"]),
        ),
        ("top_5_precision", float(metrics["top_5_precision"]) >= float(thresholds["top_5_precision"])),
        (
            "proposed_relative_vs_uniform",
            float(metrics["proposed_relative_vs_uniform"]) >= float(thresholds["proposed_relative_vs_uniform"]),
        ),
        (
            "proposed_vs_best_ablation",
            bool(metrics["proposed_ge_best_ablation"]),
        ),
        (
            "probe_allocation_overhead_ratio",
            float(metrics["probe_allocation_overhead_ratio"]) <= float(thresholds["probe_allocation_overhead_ratio"]),
        ),
    ]
    failed = [name for name, passed in checks if not passed]
    if failed:
        return "VALID_NO_GO", "Threshold failures: " + ", ".join(failed)
    return "VALID_GO", "All thresholds passed."


def main() -> int:
    args = parse_args()
    started_at = time.monotonic()
    config = _load_config(args.config)
    probe_dir = Path(args.probe_dir).expanduser().resolve()
    surrogate_dir = Path(args.surrogate_dir).expanduser().resolve()
    allocation_dir = Path(args.allocation_dir).expanduser().resolve()
    evaluation_dir = Path(args.evaluation_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = deep_update(
        config,
        {
            "paths": {
                "repo_root": str(REPO_ROOT),
                "probe_dir": str(probe_dir),
                "surrogate_dir": str(surrogate_dir),
                "allocation_dir": str(allocation_dir),
                "evaluation_dir": str(evaluation_dir),
                "output_dir": str(output_dir),
            }
        },
    )
    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(resolved_config), resolved_config_path)

    thresholds = dict(config["thresholds"])
    required_backends = [str(value) for value in config["policies"]["required_backends"]]
    configured_primary_metric = str(config["policies"]["primary_metric"])
    forbidden_tokens = [str(value) for value in config["policies"]["forbidden_path_tokens"]]
    invalid_reasons: list[str] = []
    source_run_dirs: list[str] = []

    for path_value in (probe_dir, surrogate_dir, allocation_dir, evaluation_dir):
        if path_contains_forbidden_token(path_value, forbidden_tokens):
            invalid_reasons.append(f"Forbidden mock/smoke token in source path: {path_value}")

    probe_summary = _read_json(probe_dir / "probe_summary.json")
    probe_payload = _read_json(probe_dir / "cell_utility.json")
    surrogate_summary = _read_json(surrogate_dir / "surrogate_summary.json")
    allocation_summary = _read_json(allocation_dir / "allocation_summary.json")
    evaluation_summary = _read_json(evaluation_dir / "evaluation_summary.json")
    baseline_rows = _load_long_form_rows(evaluation_dir)

    metric_names = evaluation_summary.get("metric_names")
    evaluation_metric_names = [str(value) for value in metric_names] if isinstance(metric_names, list) else []
    if configured_primary_metric != SCORE_METRIC_NAME:
        invalid_reasons.append(
            f"primary_metric policy must remain {SCORE_METRIC_NAME!r}, found {configured_primary_metric!r}"
        )
    if SCORE_METRIC_NAME not in evaluation_metric_names:
        invalid_reasons.append(f"primary_metric '{SCORE_METRIC_NAME}' not found in evaluation metric_names")

    backends_present = {str(row.get("backend")) for row in baseline_rows}
    missing_backends = [backend for backend in required_backends if backend not in backends_present]
    if missing_backends:
        invalid_reasons.append(f"Missing required baselines: {missing_backends}")

    for run_dir_value in evaluation_summary.get("evaluated_run_dirs", []):
        run_dir = Path(run_dir_value).expanduser().resolve()
        source_run_dirs.append(str(run_dir))
        if path_contains_forbidden_token(run_dir, forbidden_tokens):
            invalid_reasons.append(f"Forbidden mock/smoke token in evaluated run: {run_dir}")
            continue
        provenance = load_run_provenance(run_dir / "run_provenance.json")
        try:
            assert_real_gpu_provenance(provenance, require_gpu=False, forbid_mock=True)
        except Exception as exc:
            invalid_reasons.append(f"Non-real input run: {run_dir}: {exc}")
        for source_path in iter_recorded_source_paths(provenance):
            source_run_dirs.append(str(source_path))
            if path_contains_forbidden_token(source_path, forbidden_tokens):
                invalid_reasons.append(f"Forbidden mock/smoke token in recorded source path: {source_path}")

    aggregate_scores: dict[str, float | None] = {}
    for backend in required_backends:
        score = _extract_metric(
            baseline_rows,
            backend=backend,
            task=AGGREGATE_TASK,
            metric_name=SCORE_METRIC_NAME,
        )
        if score is None:
            invalid_reasons.append(f"aggregate score missing for required backend '{backend}'")
        aggregate_scores[backend] = score

    uniform_metric = aggregate_scores["uniform"] if "uniform" in aggregate_scores else None
    layer_only_metric = aggregate_scores["layer_only"] if "layer_only" in aggregate_scores else None
    timestep_only_metric = aggregate_scores["timestep_only"] if "timestep_only" in aggregate_scores else None
    proposed_metric = aggregate_scores["proposed"] if "proposed" in aggregate_scores else None

    best_ablation_metric = (
        max(layer_only_metric, timestep_only_metric)
        if layer_only_metric is not None and timestep_only_metric is not None
        else None
    )

    proposed_relative_vs_uniform = None
    if proposed_metric is not None and uniform_metric is not None:
        proposed_relative_vs_uniform = round(
            (proposed_metric - uniform_metric) / max(1.0e-8, abs(uniform_metric)),
            6,
        )

    overhead_numerator = (
        float(probe_summary.get("wall_time_seconds", 0.0))
        + float(surrogate_summary.get("wall_time_seconds", 0.0))
        + float(allocation_summary.get("wall_time_seconds", 0.0))
    )
    uniform_wall_subject = _extract_metric(
        baseline_rows,
        backend="uniform",
        task="subject_personalization",
        metric_name="wall_time_sec",
    )
    uniform_wall_style = _extract_metric(
        baseline_rows,
        backend="uniform",
        task="style_domain",
        metric_name="wall_time_sec",
    )
    if uniform_wall_subject is not None and uniform_wall_style is not None:
        denominator = uniform_wall_subject + uniform_wall_style
        if denominator <= 0.0:
            invalid_reasons.append("uniform wall_time_sec sum is zero")
            probe_allocation_overhead_ratio = None
        else:
            probe_allocation_overhead_ratio = round(overhead_numerator / denominator, 6)
    else:
        invalid_reasons.append("uniform wall_time_sec missing for one or both tasks")
        probe_allocation_overhead_ratio = None

    metrics = {
        "top_20_mass_ratio": _top_20_mass_ratio(probe_payload.get("rows", [])),
        "held_out_spearman_rho": float(surrogate_summary["held_out_spearman_rho"]),
        "top_5_precision": float(surrogate_summary["top_5_precision"]),
        "proposed_relative_vs_uniform": proposed_relative_vs_uniform,
        "proposed_metric": proposed_metric,
        "uniform_metric": uniform_metric,
        "best_ablation_metric": best_ablation_metric,
        "proposed_ge_best_ablation": (
            proposed_metric >= best_ablation_metric
            if proposed_metric is not None and best_ablation_metric is not None
            else None
        ),
        "probe_allocation_overhead_ratio": probe_allocation_overhead_ratio,
    }
    gate_status, decision_summary = _status_and_summary(
        metrics=metrics,
        thresholds=thresholds,
        invalid_reasons=invalid_reasons,
    )

    memo_json_path = output_dir / "gate_memo.json"
    memo_md_path = output_dir / "gate_memo.md"
    payload = {
        "gate_status": gate_status,
        "thresholds": thresholds,
        "metrics": metrics,
        "source_run_dirs": sorted(set(source_run_dirs)),
        "invalid_reasons": invalid_reasons,
        "decision_summary": decision_summary,
        "wall_time_seconds": round(time.monotonic() - started_at, 6),
    }
    save_json(memo_json_path, payload)
    memo_md_path.write_text(
        "\n".join(
            [
                "# RD-LoRA Gate Memo",
                "",
                f"Status: **{gate_status}**",
                "",
                f"Summary: {decision_summary}",
                "",
                "## Invalid Reasons",
                *(["- None"] if not invalid_reasons else [f"- {reason}" for reason in invalid_reasons]),
                "",
                "## Metrics",
                *(f"- {key}: {value}" for key, value in metrics.items()),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"resolved_config={resolved_config_path}")
    print(f"gate_memo_json={memo_json_path}")
    print(f"gate_memo_md={memo_md_path}")
    print(f"gate_status={gate_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
