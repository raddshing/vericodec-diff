from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


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
        description="Evaluate RD-LoRA train runs and emit the standardized evaluation artifacts."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config path.")
    parser.add_argument(
        "--run_dir",
        action="append",
        nargs="+",
        default=[],
        help="Repeatable train run directory argument. Each use may include one or more directories.",
    )
    parser.add_argument("--output_dir", required=True, help="Output directory for evaluation artifacts.")
    return parser.parse_args()


def _default_config() -> dict[str, Any]:
    return {"evaluation": {"primary_metric": "score"}}


def _load_config(path_value: str) -> dict[str, Any]:
    config_path = Path(path_value).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    raw = load_yaml_mapping(config_path) if config_path.is_file() else {}
    return deep_update(_default_config(), raw)


def _numeric_metric_names(rows: list[dict[str, Any]]) -> list[str]:
    metric_names: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key in {"backend", "task", "run_dir"}:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                metric_names.append(key)
    return sorted(set(metric_names))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> int:
    args = parse_args()
    started_at = time.monotonic()
    if not args.run_dir:
        raise SystemExit("At least one --run_dir is required")

    config = _load_config(args.config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [item for group in args.run_dir for item in group]
    resolved_config = deep_update(
        config,
        {
            "paths": {
                "repo_root": str(REPO_ROOT),
                "output_dir": str(output_dir),
                "run_dirs": [str(Path(value).expanduser().resolve()) for value in run_dirs],
            }
        },
    )
    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(resolved_config), resolved_config_path)

    rows: list[dict[str, Any]] = []
    tasks: set[str] = set()
    backends: set[str] = set()
    evaluated_run_dirs: list[str] = []
    for run_dir_value in run_dirs:
        run_dir = Path(run_dir_value).expanduser().resolve()
        if path_contains_forbidden_token(run_dir, FORBIDDEN_SOURCE_TOKENS):
            raise RuntimeError(f"Refusing mock/smoke run_dir: {run_dir}")
        provenance = load_run_provenance(run_dir / "run_provenance.json")
        assert_real_gpu_provenance(provenance, require_gpu=False, forbid_mock=True)
        for source_path in iter_recorded_source_paths(provenance):
            if path_contains_forbidden_token(source_path, FORBIDDEN_SOURCE_TOKENS):
                raise RuntimeError(f"Refusing mock/smoke source path: {source_path}")
        train_summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        row = {
            "backend": str(train_summary["backend"]),
            "task": str(train_summary["task"]),
            "run_dir": str(run_dir),
        }
        for key, value in metrics.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                row[key] = value
        rows.append(row)
        tasks.add(row["task"])
        backends.add(row["backend"])
        evaluated_run_dirs.append(str(run_dir))

    metric_names = _numeric_metric_names(rows)
    baseline_metrics_json_path = output_dir / "baseline_metrics.json"
    baseline_metrics_csv_path = output_dir / "baseline_metrics.csv"
    evaluation_summary_path = output_dir / "evaluation_summary.json"

    save_json(
        baseline_metrics_json_path,
        {
            "schema_version": "1.0",
            "rows": rows,
        },
    )
    _write_csv(
        baseline_metrics_csv_path,
        rows,
        fieldnames=["backend", "task", "run_dir", *metric_names],
    )
    save_json(
        evaluation_summary_path,
        {
            "schema_version": "1.0",
            "tasks": sorted(tasks),
            "backends": sorted(backends),
            "metric_names": metric_names,
            "evaluated_run_dirs": evaluated_run_dirs,
            "primary_metric": str(config["evaluation"]["primary_metric"]),
            "wall_time_seconds": round(time.monotonic() - started_at, 6),
        },
    )

    print(f"resolved_config={resolved_config_path}")
    print(f"evaluation_summary={evaluation_summary_path}")
    print(f"baseline_metrics_json={baseline_metrics_json_path}")
    print(f"baseline_metrics_csv={baseline_metrics_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
