from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vericodec_diff.config import OmegaConf
from vericodec_diff.patch_metrics import deep_update
from vericodec_diff.verifier_training import (
    evaluate_verifier_models,
    resolve_verifier_eval_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained verifier checkpoints against validation and kill-test splits."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional repo-relative or absolute YAML config file for verifier evaluation.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Target root that receives outputs/metrics/kill_test/ for this run.",
    )
    parser.add_argument(
        "--feature-root",
        default=None,
        help="Repo-relative or absolute root containing data/processed/verifier_features/<split>/ files.",
    )
    parser.add_argument(
        "--error-map-root",
        default=None,
        help="Repo-relative or absolute root containing outputs/error_maps/<split>/ files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Repo-relative or absolute directory containing checkpoints/verifier/ outputs.",
    )
    parser.add_argument(
        "--checkpoint-index",
        default=None,
        help="Optional explicit checkpoint index JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Repo-relative or absolute directory receiving evaluation JSON and budget CSV outputs.",
    )
    parser.add_argument(
        "--signal-names",
        default=None,
        help="Optional comma-separated verifier signal list.",
    )
    parser.add_argument(
        "--val-splits",
        default=None,
        help="Optional comma-separated validation split list.",
    )
    parser.add_argument(
        "--test-splits",
        default=None,
        help="Optional comma-separated test split list.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for smoke runs.",
    )
    parser.add_argument(
        "--label-metric-name",
        default=None,
        help="Patch metric name used to derive binary labels.",
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=None,
        help="Threshold applied to the selected patch metric to define positives.",
    )
    parser.add_argument(
        "--budget-percents",
        default=None,
        help="Optional comma-separated patch-budget percentages for recovery curves.",
    )
    return parser.parse_args()


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must parse to a mapping")
    return data


def _build_cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {
        "paths": {
            "repo_root": args.repo_root,
        }
    }

    for arg_name, config_key in (
        ("feature_root", "feature_root"),
        ("error_map_root", "error_map_root"),
        ("checkpoint_dir", "checkpoint_dir"),
        ("checkpoint_index", "checkpoint_index"),
        ("output_dir", "output_dir"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides["paths"][config_key] = value

    data_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("signal_names", "signal_names"),
        ("val_splits", "val_splits"),
        ("test_splits", "test_splits"),
        ("limit", "limit"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            data_overrides[config_key] = value
    if data_overrides:
        overrides["data"] = data_overrides

    label_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("label_metric_name", "metric_name"),
        ("label_threshold", "threshold"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            label_overrides[config_key] = value
    if label_overrides:
        overrides["labels"] = label_overrides

    if args.budget_percents is not None:
        overrides["evaluation"] = {"budget_percents": args.budget_percents}

    return overrides


def main() -> int:
    args = parse_args()
    raw_config: dict[str, object] = {}
    if args.config is not None:
        config_path = Path(args.config).expanduser()
        if not config_path.is_absolute():
            config_path = (REPO_ROOT / config_path).resolve()
        raw_config = _load_yaml_mapping(config_path)

    raw_config = deep_update(raw_config, _build_cli_overrides(args))
    config = resolve_verifier_eval_config(REPO_ROOT, raw_config)

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = OmegaConf.create(config)
    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(resolved_config, resolved_config_path)

    result = evaluate_verifier_models(config)
    payload = dict(result["payload"])
    best_model = dict(payload["best_model"])
    heuristic = dict(payload["heuristic_baseline"])

    print(f"resolved_config={resolved_config_path}")
    print(f"eval_json={result['eval_json_path']}")
    print(f"budget_curves_csv={result['budget_curve_path']}")
    print(f"output_dir={output_dir}")
    print(f"best_checkpoint={best_model['checkpoint_filename']}")
    print(f"best_model_type={best_model['model_type']}")
    print(f"best_model_test_auprc={best_model['test_metrics']['auprc']}")
    print(f"best_model_test_auroc={best_model['test_metrics']['auroc']}")
    print(f"heuristic_signal={heuristic['signal_name']}")
    print(f"heuristic_direction={heuristic['direction_name']}")
    print(f"heuristic_test_auprc={heuristic['test_metrics']['auprc']}")
    print(f"heuristic_test_auroc={heuristic['test_metrics']['auroc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
