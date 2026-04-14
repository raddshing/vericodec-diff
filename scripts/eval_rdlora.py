from __future__ import annotations

import argparse
import csv
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

import rd_lora.eval.generate as generate_module
from rd_lora.eval.generate import generate_eval_images_for_run
from rd_lora.eval.metrics_style import compute_style_metrics
from rd_lora.eval.metrics_subject import compute_subject_metrics
from rd_lora.eval.score import backend_score
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
DEFAULT_SUBJECT_MANIFEST = "data/rd_lora/eval/subject_personalization_eval.csv"
DEFAULT_STYLE_MANIFEST = "data/rd_lora/eval/style_domain_eval.csv"
DEFAULT_REFERENCE_IMAGE_DIR = "tests/fixtures/rdlora_pilot"
DEFAULT_BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_DEVICE = "cuda:0"
DEFAULT_OCR_ENGINE = "tesseract"
PRIMARY_METRIC = "score"
AGGREGATE_TASK = "__aggregate__"
SUPPORTED_TASKS = ("subject_personalization", "style_domain")
LONG_FORM_COLUMNS = [
    "task",
    "backend",
    "metric_name",
    "metric_value",
    "run_dir",
    "source_json",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RD-LoRA runs by validating provenance, generating held-out images, and emitting long-form baseline metrics."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config path.")
    parser.add_argument(
        "--run_dir",
        action="append",
        nargs="+",
        default=[],
        help="Repeatable run directory argument. Each use may include one or more run directories.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for evaluation artifacts.")
    parser.add_argument(
        "--subject_manifest",
        default=DEFAULT_SUBJECT_MANIFEST,
        help="CSV manifest for subject_personalization evaluation prompts.",
    )
    parser.add_argument(
        "--style_manifest",
        default=DEFAULT_STYLE_MANIFEST,
        help="CSV manifest for style_domain evaluation prompts.",
    )
    parser.add_argument(
        "--reference_image_dir",
        default=DEFAULT_REFERENCE_IMAGE_DIR,
        help="Reference image directory for subject_personalization metrics.",
    )
    parser.add_argument(
        "--base_model_id",
        default=DEFAULT_BASE_MODEL_ID,
        help="Base SDXL model id used to load the eval generation pipeline.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help="Classifier-free guidance scale for held-out image generation.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help="Number of diffusion denoising steps used during evaluation generation.",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Torch device for generation and metrics.")
    parser.add_argument("--ocr_engine", default=DEFAULT_OCR_ENGINE, help="OCR engine for style_domain metrics.")
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Assume held-out images already exist in cache and reuse generation_records.csv.",
    )
    return parser.parse_args(argv)


def _default_config() -> dict[str, Any]:
    return {"evaluation": {"primary_metric": PRIMARY_METRIC}}


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _flatten_run_dirs(run_dir_groups: Sequence[Sequence[str]]) -> list[Path]:
    flattened: list[Path] = []
    for group in run_dir_groups:
        for value in group:
            flattened.append(Path(value).expanduser().resolve())
    return flattened


def _load_config(path_value: str) -> tuple[Path, dict[str, Any]]:
    config_path = _resolve_repo_path(path_value)
    raw = load_yaml_mapping(config_path) if config_path.is_file() else {}
    return config_path, deep_update(_default_config(), raw)


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _require_non_empty_string(payload: Mapping[str, Any], key: str, source_path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{source_path} is missing non-empty string field {key!r}")
    return value


def _resolve_checkpoint_path(run_dir: Path, checkpoint_value: str) -> Path:
    checkpoint_path = Path(checkpoint_value).expanduser()
    if checkpoint_path.is_absolute():
        return checkpoint_path.resolve()
    return (run_dir / checkpoint_path).resolve()


def _iter_provenance_paths(payload: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("diffusers_file", "accelerate_config_file", "allocation_manifest"):
        value = payload.get(key)
        if value not in (None, ""):
            paths.append(str(value))
    paths.extend(iter_recorded_source_paths(payload))
    return paths


def _assert_no_forbidden_path(path_value: str | Path, *, label: str) -> None:
    if path_contains_forbidden_token(path_value, FORBIDDEN_SOURCE_TOKENS):
        raise RuntimeError(f"Forbidden mock/smoke token found in {label}: {path_value}")


def _verify_real_gpu_provenance(run_dir: Path) -> None:
    _assert_no_forbidden_path(run_dir, label="run_dir")
    provenance = load_run_provenance(run_dir / "run_provenance.json")
    assert_real_gpu_provenance(
        provenance,
        require_gpu=True,
        forbid_mock=True,
    )
    for recorded_path in _iter_provenance_paths(provenance):
        _assert_no_forbidden_path(recorded_path, label="recorded provenance path")


def _load_train_summary(run_dir: Path) -> tuple[dict[str, Any], str, str, Path]:
    train_summary_path = run_dir / "train_summary.json"
    train_summary = _read_json_mapping(train_summary_path)
    backend = _require_non_empty_string(train_summary, "backend", train_summary_path)
    task = _require_non_empty_string(train_summary, "task", train_summary_path)
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task {task!r} in {train_summary_path}")
    return train_summary, backend, task, train_summary_path


def _locate_checkpoint_path(
    run_dir: Path,
    *,
    train_summary: Mapping[str, Any],
    train_summary_path: Path,
) -> tuple[Path, Path]:
    checkpoint_info_path = run_dir / "checkpoint_info.json"
    if checkpoint_info_path.is_file():
        checkpoint_info = _read_json_mapping(checkpoint_info_path)
        checkpoint_value = _require_non_empty_string(checkpoint_info, "checkpoint_path", checkpoint_info_path)
        source_json = checkpoint_info_path
    else:
        checkpoint_value = _require_non_empty_string(train_summary, "checkpoint_path", train_summary_path)
        source_json = train_summary_path

    checkpoint_path = _resolve_checkpoint_path(run_dir, checkpoint_value)
    _assert_no_forbidden_path(checkpoint_path, label="checkpoint_path")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
    return checkpoint_path, source_json


def _manifest_path_for_task(
    task: str,
    *,
    subject_manifest: Path,
    style_manifest: Path,
) -> Path:
    if task == "subject_personalization":
        return subject_manifest
    if task == "style_domain":
        return style_manifest
    raise ValueError(f"Unsupported task {task!r}")


def _generate_records_for_run(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    cache_dir: Path,
    manifest_path: Path,
    base_model_id: str,
    guidance_scale: float,
    num_inference_steps: int,
    device: str,
    skip_generation: bool,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    records_path = cache_dir / "generation_records.csv"
    if skip_generation:
        if not records_path.is_file():
            raise FileNotFoundError(
                f"--skip_generation was set but generation_records.csv is missing: {records_path}"
            )
        return records_path

    checkpoint_info_path = (run_dir / "checkpoint_info.json").resolve()
    if checkpoint_info_path.is_file():
        return generate_eval_images_for_run(
            run_dir=run_dir,
            prompt_manifest_path=manifest_path,
            output_dir=cache_dir,
            base_model_id=base_model_id,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            device=device,
            torch_dtype="float16",
        )

    original_load_json_mapping = generate_module._load_json_mapping

    def _patched_load_json_mapping(path: Path) -> dict[str, Any]:
        if Path(path).expanduser().resolve() == checkpoint_info_path:
            return {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_exists": True,
            }
        return original_load_json_mapping(path)

    generate_module._load_json_mapping = _patched_load_json_mapping
    try:
        return generate_eval_images_for_run(
            run_dir=run_dir,
            prompt_manifest_path=manifest_path,
            output_dir=cache_dir,
            base_model_id=base_model_id,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            device=device,
            torch_dtype="float16",
        )
    finally:
        generate_module._load_json_mapping = original_load_json_mapping


def _write_metrics_json(path: Path, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(metrics), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _compute_metrics_for_run(
    *,
    task: str,
    generated_records_csv: Path,
    reference_image_dir: Path,
    device: str,
    ocr_engine: str,
) -> dict[str, Any]:
    if task == "subject_personalization":
        if not reference_image_dir.is_dir():
            raise FileNotFoundError(f"reference_image_dir does not exist: {reference_image_dir}")
        return compute_subject_metrics(
            generated_records_csv=generated_records_csv,
            reference_image_dir=reference_image_dir,
            device=device,
        )
    if task == "style_domain":
        return compute_style_metrics(
            generated_records_csv=generated_records_csv,
            device=device,
            ocr_engine=ocr_engine,
        )
    raise ValueError(f"Unsupported task {task!r}")


def _coerce_optional_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isfinite(numeric):
            return numeric
    return None


def _append_metric_rows(
    rows: list[dict[str, Any]],
    *,
    task: str,
    backend: str,
    metrics: Mapping[str, Any],
    run_dir: Path | None,
    source_json: Path,
) -> None:
    run_dir_value = "" if run_dir is None else str(run_dir)
    for metric_name, raw_value in metrics.items():
        if metric_name == "num_images":
            continue
        metric_value = _coerce_optional_number(raw_value)
        if metric_value is None:
            raise ValueError(f"Metric {metric_name!r} for backend={backend} task={task} is not numeric")
        rows.append(
            {
                "task": task,
                "backend": backend,
                "metric_name": str(metric_name),
                "metric_value": metric_value,
                "run_dir": run_dir_value,
                "source_json": str(source_json),
            }
        )


def _append_wall_time_row(
    rows: list[dict[str, Any]],
    *,
    task: str,
    backend: str,
    run_dir: Path,
    train_summary: Mapping[str, Any],
    train_summary_path: Path,
) -> None:
    wall_time_sec = _coerce_optional_number(train_summary.get("wall_time_sec"))
    if wall_time_sec is None:
        return
    rows.append(
        {
            "task": task,
            "backend": backend,
            "metric_name": "wall_time_sec",
            "metric_value": wall_time_sec,
            "run_dir": str(run_dir),
            "source_json": str(train_summary_path),
        }
    )


def _build_score_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    scores_by_backend: dict[str, dict[str, float]] = {}
    for row in rows:
        task = str(row["task"])
        if task == AGGREGATE_TASK or str(row["metric_name"]) != PRIMARY_METRIC:
            continue
        backend = str(row["backend"])
        backend_scores = scores_by_backend.setdefault(backend, {})
        if task in backend_scores:
            raise ValueError(f"Duplicate {PRIMARY_METRIC!r} rows for backend={backend} task={task}")
        backend_scores[task] = float(row["metric_value"])
    return scores_by_backend


def _build_wall_time_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    wall_times_by_backend: dict[str, dict[str, float]] = {}
    for row in rows:
        task = str(row["task"])
        if task == AGGREGATE_TASK or str(row["metric_name"]) != "wall_time_sec":
            continue
        backend = str(row["backend"])
        backend_wall_times = wall_times_by_backend.setdefault(backend, {})
        if task in backend_wall_times:
            raise ValueError(f"Duplicate 'wall_time_sec' rows for backend={backend} task={task}")
        backend_wall_times[task] = float(row["metric_value"])
    return wall_times_by_backend


def _append_aggregate_rows(rows: list[dict[str, Any]], *, aggregate_source_json: Path) -> None:
    scores_by_backend = _build_score_map(rows)
    wall_times_by_backend = _build_wall_time_map(rows)
    for backend in sorted(scores_by_backend):
        try:
            aggregate_score = backend_score(scores_by_backend[backend])
        except ValueError as exc:
            raise ValueError(f"Unable to compute aggregate {PRIMARY_METRIC!r} for backend={backend}: {exc}") from exc

        rows.append(
            {
                "task": AGGREGATE_TASK,
                "backend": backend,
                "metric_name": PRIMARY_METRIC,
                "metric_value": aggregate_score,
                "run_dir": "",
                "source_json": str(aggregate_source_json),
            }
        )

        task_wall_times = wall_times_by_backend.get(backend, {})
        task_scores = scores_by_backend[backend]
        if not all(task in task_wall_times for task in task_scores):
            continue
        rows.append(
            {
                "task": AGGREGATE_TASK,
                "backend": backend,
                "metric_name": "wall_time_sec",
                "metric_value": round(sum(task_wall_times[task] for task in task_scores), 6),
                "run_dir": "",
                "source_json": str(aggregate_source_json),
            }
        )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LONG_FORM_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in LONG_FORM_COLUMNS})


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started_at = time.monotonic()
    if not args.run_dir:
        raise SystemExit("At least one --run_dir is required")

    config_path, config = _load_config(args.config)
    run_dirs = _flatten_run_dirs(args.run_dir)
    if not run_dirs:
        raise SystemExit("At least one --run_dir is required")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_manifest = _resolve_repo_path(args.subject_manifest)
    style_manifest = _resolve_repo_path(args.style_manifest)
    reference_image_dir = _resolve_repo_path(args.reference_image_dir)

    resolved_config = deep_update(
        config,
        {
            "config_path": str(config_path),
            "paths": {
                "repo_root": str(REPO_ROOT),
                "output_dir": str(output_dir),
                "run_dirs": [str(path) for path in run_dirs],
                "subject_manifest": str(subject_manifest),
                "style_manifest": str(style_manifest),
                "reference_image_dir": str(reference_image_dir),
            },
            "evaluation": {
                "primary_metric": PRIMARY_METRIC,
                "base_model_id": args.base_model_id,
                "guidance_scale": float(args.guidance_scale),
                "num_inference_steps": int(args.num_inference_steps),
                "device": str(args.device),
                "ocr_engine": str(args.ocr_engine),
                "skip_generation": bool(args.skip_generation),
            },
        },
    )
    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(resolved_config), resolved_config_path)

    baseline_metrics_json_path = output_dir / "baseline_metrics.json"
    baseline_metrics_csv_path = output_dir / "baseline_metrics.csv"
    evaluation_summary_path = output_dir / "evaluation_summary.json"

    rows: list[dict[str, Any]] = []
    tasks: set[str] = set()
    backends: set[str] = set()
    evaluated_run_dirs: list[str] = []

    for run_dir in run_dirs:
        _verify_real_gpu_provenance(run_dir)
        train_summary, backend, task, train_summary_path = _load_train_summary(run_dir)
        checkpoint_path, _ = _locate_checkpoint_path(
            run_dir,
            train_summary=train_summary,
            train_summary_path=train_summary_path,
        )

        cache_dir = output_dir / "generated" / f"{backend}__{task}"
        manifest_path = _manifest_path_for_task(
            task,
            subject_manifest=subject_manifest,
            style_manifest=style_manifest,
        )
        generated_records_csv = _generate_records_for_run(
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            cache_dir=cache_dir,
            manifest_path=manifest_path,
            base_model_id=args.base_model_id,
            guidance_scale=float(args.guidance_scale),
            num_inference_steps=int(args.num_inference_steps),
            device=str(args.device),
            skip_generation=bool(args.skip_generation),
        )

        metrics = _compute_metrics_for_run(
            task=task,
            generated_records_csv=generated_records_csv,
            reference_image_dir=reference_image_dir,
            device=str(args.device),
            ocr_engine=str(args.ocr_engine),
        )
        metrics_json_path = cache_dir / "metrics.json"
        _write_metrics_json(metrics_json_path, metrics)

        _append_metric_rows(
            rows,
            task=task,
            backend=backend,
            metrics=metrics,
            run_dir=run_dir,
            source_json=metrics_json_path,
        )
        _append_wall_time_row(
            rows,
            task=task,
            backend=backend,
            run_dir=run_dir,
            train_summary=train_summary,
            train_summary_path=train_summary_path,
        )

        tasks.add(task)
        backends.add(backend)
        evaluated_run_dirs.append(str(run_dir))

    _append_aggregate_rows(rows, aggregate_source_json=baseline_metrics_json_path)
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            str(row["backend"]),
            str(row["task"]),
            str(row["metric_name"]),
            str(row["run_dir"]),
            str(row["source_json"]),
        ),
    )

    _write_csv(baseline_metrics_csv_path, sorted_rows)
    save_json(
        baseline_metrics_json_path,
        {
            "schema_version": "1.0",
            "rows": sorted_rows,
        },
    )
    save_json(
        evaluation_summary_path,
        {
            "schema_version": "1.0",
            "tasks": sorted(tasks),
            "backends": sorted(backends),
            "metric_names": sorted({str(row["metric_name"]) for row in sorted_rows}),
            "evaluated_run_dirs": evaluated_run_dirs,
            "primary_metric": PRIMARY_METRIC,
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
