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
from vericodec_diff.patch_metrics import deep_update, display_path
from vericodec_diff.verifier_training import (
    train_verifier_models,
    resolve_verifier_train_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train patchwise verifier models on processed verifier features and patch labels."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional repo-relative or absolute YAML config file for verifier training.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Target root that receives checkpoints/ for this verifier-training run.",
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
        help="Repo-relative or absolute directory receiving checkpoints/verifier/ outputs.",
    )
    parser.add_argument(
        "--signal-names",
        default=None,
        help="Optional comma-separated verifier signal list.",
    )
    parser.add_argument(
        "--train-splits",
        default=None,
        help="Optional comma-separated train split list.",
    )
    parser.add_argument(
        "--val-splits",
        default=None,
        help="Optional comma-separated validation split list.",
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
        "--models-enabled",
        default=None,
        help="Optional comma-separated model list drawn from logistic,mlp.",
    )
    parser.add_argument(
        "--models-seed",
        type=int,
        default=None,
        help="Random seed used by the verifier MLP initialization.",
    )
    parser.add_argument(
        "--logistic-learning-rate",
        type=float,
        default=None,
        help="Learning rate for the logistic verifier.",
    )
    parser.add_argument(
        "--logistic-epochs",
        type=int,
        default=None,
        help="Epoch count for the logistic verifier.",
    )
    parser.add_argument(
        "--logistic-l2",
        type=float,
        default=None,
        help="L2 penalty for the logistic verifier.",
    )
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension for the tiny 2-layer MLP verifier.",
    )
    parser.add_argument(
        "--mlp-learning-rate",
        type=float,
        default=None,
        help="Learning rate for the tiny 2-layer MLP verifier.",
    )
    parser.add_argument(
        "--mlp-epochs",
        type=int,
        default=None,
        help="Epoch count for the tiny 2-layer MLP verifier.",
    )
    parser.add_argument(
        "--mlp-l2",
        type=float,
        default=None,
        help="L2 penalty for the tiny 2-layer MLP verifier.",
    )
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow overwriting existing verifier artifacts with deterministic names.",
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
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides["paths"][config_key] = value

    data_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("signal_names", "signal_names"),
        ("train_splits", "train_splits"),
        ("val_splits", "val_splits"),
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

    model_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("models_enabled", "enabled"),
        ("models_seed", "seed"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            model_overrides[config_key] = value

    logistic_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("logistic_learning_rate", "learning_rate"),
        ("logistic_epochs", "epochs"),
        ("logistic_l2", "l2"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            logistic_overrides[config_key] = value
    if logistic_overrides:
        model_overrides["logistic"] = logistic_overrides

    mlp_overrides: dict[str, object] = {}
    for arg_name, config_key in (
        ("mlp_hidden_dim", "hidden_dim"),
        ("mlp_learning_rate", "learning_rate"),
        ("mlp_epochs", "epochs"),
        ("mlp_l2", "l2"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            mlp_overrides[config_key] = value
    if mlp_overrides:
        model_overrides["mlp"] = mlp_overrides

    if model_overrides:
        overrides["models"] = model_overrides

    if args.overwrite is not None:
        overrides["runtime"] = {"overwrite": args.overwrite}

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
    config = resolve_verifier_train_config(REPO_ROOT, raw_config)

    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = OmegaConf.create(config)
    resolved_config_path = checkpoint_dir / "resolved_config.yaml"
    OmegaConf.save(resolved_config, resolved_config_path)

    result = train_verifier_models(config)

    repo_root = Path(config["paths"]["repo_root"])
    best_checkpoint = dict(result["best_checkpoint"])
    print(f"resolved_config={resolved_config_path}")
    print(f"checkpoint_index={result['checkpoint_index_path']}")
    print(f"training_summary={result['training_summary_path']}")
    print(f"checkpoint_dir={checkpoint_dir}")
    print(f"train_samples={result['train_dataset']['sample_count']}")
    print(f"val_samples={result['val_dataset']['sample_count']}")
    print(f"best_checkpoint={best_checkpoint['checkpoint_filename']}")
    print(f"best_model_type={best_checkpoint['model_type']}")
    print(f"best_val_auprc={best_checkpoint['val_metrics']['auprc']}")
    print(f"best_val_auroc={best_checkpoint['val_metrics']['auroc']}")
    print(f"best_checkpoint_path={display_path(Path(best_checkpoint['checkpoint_path']), repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
