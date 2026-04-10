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


def _primary_metric_rows(evaluation_dir: Path) -> list[dict[str, Any]]:
    payload = _read_json(evaluation_dir / "baseline_metrics.json")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("baseline_metrics.json must contain rows")
    return [dict(row) for row in rows]


def _wall_time_for_backend(rows: Sequence[Mapping[str, Any]], backend: str) -> float | None:
    for row in rows:
        if str(row["backend"]) == backend:
            value = row.get("wall_time_seconds")
            if value in (None, ""):
                return None
            return float(value)
    return None


def _status_and_summary(
    *,
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    invalid_reasons: list[str],
) -> tuple[str, str]:
    if invalid_reasons:
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
    baseline_rows = _primary_metric_rows(evaluation_dir)

    backends_present = {str(row["backend"]) for row in baseline_rows}
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

    baseline_by_backend = {str(row["backend"]): dict(row) for row in baseline_rows}
    primary_metric = str(config["policies"]["primary_metric"])
    uniform_metric = float(baseline_by_backend.get("uniform", {}).get(primary_metric, 0.0))
    proposed_metric = float(baseline_by_backend.get("proposed", {}).get(primary_metric, 0.0))
    best_ablation_metric = max(
        float(baseline_by_backend.get("layer_only", {}).get(primary_metric, 0.0)),
        float(baseline_by_backend.get("timestep_only", {}).get(primary_metric, 0.0)),
    )
    denominator = max(abs(uniform_metric), 1.0e-8)
    proposed_relative_vs_uniform = round((proposed_metric - uniform_metric) / denominator, 6)
    uniform_wall_time = _wall_time_for_backend(baseline_rows, "uniform")
    overhead_numerator = float(probe_summary.get("wall_time_seconds", 0.0)) + float(
        surrogate_summary.get("wall_time_seconds", 0.0)
    ) + float(allocation_summary.get("wall_time_seconds", 0.0))
    probe_allocation_overhead_ratio = (
        round(overhead_numerator / float(uniform_wall_time), 6)
        if uniform_wall_time not in (None, 0.0)
        else float("inf")
    )

    metrics = {
        "top_20_mass_ratio": _top_20_mass_ratio(probe_payload.get("rows", [])),
        "held_out_spearman_rho": float(surrogate_summary["held_out_spearman_rho"]),
        "top_5_precision": float(surrogate_summary["top_5_precision"]),
        "proposed_relative_vs_uniform": proposed_relative_vs_uniform,
        "proposed_metric": proposed_metric,
        "uniform_metric": uniform_metric,
        "best_ablation_metric": best_ablation_metric,
        "proposed_ge_best_ablation": proposed_metric >= best_ablation_metric,
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
