from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.eval.metrics_subject import compute_subject_metrics
from rd_lora.eval.metrics_style import compute_style_metrics
from vericodec_diff.config import OmegaConf


DEFAULT_CONFIG_PATH = "configs/rdlora_gate.yaml"
SUPPORTED_TASKS = ("subject_personalization", "style_domain")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-run RD-LoRA evaluation metrics for one training run."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config path.")
    parser.add_argument("--run_dir", required=True, help="Path to a single training run directory.")
    parser.add_argument(
        "--task",
        choices=SUPPORTED_TASKS,
        required=True,
        help="Evaluation task to score for the provided run.",
    )
    parser.add_argument(
        "--generated_records_csv",
        required=True,
        help="Path to generation_records.csv emitted by the eval image generation step.",
    )
    parser.add_argument("--output_json", required=True, help="Path to write the per-run metrics JSON.")
    parser.add_argument(
        "--reference_image_dir",
        default=None,
        help="Reference image directory for subject_personalization. Ignored for style_domain.",
    )
    parser.add_argument("--device", default="cuda:0", help="GPU device for metric encoders.")
    parser.add_argument("--ocr_engine", default="tesseract", help="OCR engine for style_domain.")
    return parser.parse_args(argv)


def _fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _resolve_config_path(config_value: str) -> Path:
    config_path = Path(config_value).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    else:
        config_path = config_path.resolve()
    return config_path


def _resolve_path(path_value: str) -> Path:
    return Path(path_value).expanduser().resolve()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"config not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        _fail(f"malformed config YAML at {path}: {exc}")
    except OSError as exc:
        _fail(f"unable to read config at {path}: {exc}")

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        _fail(f"config must contain a YAML mapping: {path}")
    return payload


def _require_existing_file(path: Path, label: str) -> None:
    if not path.is_file():
        _fail(f"{label} not found: {path}")


def _require_existing_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        _fail(f"{label} not found: {path}")


def _load_train_summary(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        _fail(f"malformed train_summary.json at {path}: {exc.msg}")
    except OSError as exc:
        _fail(f"unable to read train_summary.json at {path}: {exc}")

    if not isinstance(payload, dict):
        _fail(f"malformed train_summary.json at {path}: expected a JSON object")

    backend = payload.get("backend")
    task = payload.get("task")
    if not isinstance(backend, str) or backend.strip() == "":
        _fail(f"malformed train_summary.json at {path}: missing non-empty string field 'backend'")
    if not isinstance(task, str) or task.strip() == "":
        _fail(f"malformed train_summary.json at {path}: missing non-empty string field 'task'")
    return payload


def _write_resolved_config(
    resolved_config_path: Path,
    *,
    config_path: Path,
    config: dict[str, Any],
    run_dir: Path,
    task: str,
    generated_records_csv: Path,
    output_json: Path,
    reference_image_dir: Path | None,
    device: str,
    ocr_engine: str,
) -> None:
    resolved_config = {
        "config_path": str(config_path),
        "config": config,
        "request": {
            "run_dir": str(run_dir),
            "task": task,
            "generated_records_csv": str(generated_records_csv),
            "output_json": str(output_json),
            "reference_image_dir": str(reference_image_dir) if reference_image_dir is not None else None,
            "device": device,
            "ocr_engine": ocr_engine,
        },
    }
    OmegaConf.save(OmegaConf.create(resolved_config), resolved_config_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    config_path = _resolve_config_path(args.config)
    config = _load_yaml_mapping(config_path)

    run_dir = _resolve_path(args.run_dir)
    generated_records_csv = _resolve_path(args.generated_records_csv)
    output_json = _resolve_path(args.output_json)
    reference_image_dir = (
        _resolve_path(args.reference_image_dir) if args.reference_image_dir is not None else None
    )

    _require_existing_file(generated_records_csv, "generated_records_csv")
    _require_existing_dir(run_dir, "run_dir")
    train_summary_path = run_dir / "train_summary.json"
    _require_existing_file(train_summary_path, "train_summary.json")
    train_summary = _load_train_summary(train_summary_path)

    backend = str(train_summary["backend"])
    summary_task = str(train_summary["task"])
    if args.task != summary_task:
        _fail(f"--task={args.task} does not match train_summary.json task={summary_task}")

    if output_json.exists() and output_json.is_dir():
        _fail(f"output_json must be a file path: {output_json}")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_json.parent / "resolved_config.yaml"
    _write_resolved_config(
        resolved_config_path,
        config_path=config_path,
        config=config,
        run_dir=run_dir,
        task=args.task,
        generated_records_csv=generated_records_csv,
        output_json=output_json,
        reference_image_dir=reference_image_dir,
        device=args.device,
        ocr_engine=args.ocr_engine,
    )

    if args.task == "subject_personalization":
        if reference_image_dir is None:
            _fail("--reference_image_dir is required for subject_personalization")
        _require_existing_dir(reference_image_dir, "reference_image_dir")
        metrics_result = compute_subject_metrics(
            generated_records_csv=generated_records_csv,
            reference_image_dir=reference_image_dir,
            device=args.device,
        )
    else:
        metrics_result = compute_style_metrics(
            generated_records_csv=generated_records_csv,
            device=args.device,
            ocr_engine=args.ocr_engine,
        )

    payload: dict[str, Any] = {
        "run_dir": str(run_dir),
        "backend": backend,
        "task": args.task,
        "num_images": int(metrics_result["num_images"]),
    }
    for key, value in metrics_result.items():
        if key in {"num_images", "score"}:
            continue
        payload[key] = value
    payload["score"] = float(metrics_result["score"])

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"metrics_json={output_json}")
    print(f"task={args.task}")
    print(f"backend={backend}")
    print(f"score={payload['score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
