from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vericodec_diff.patch_metrics import (
    DEFAULT_GRID_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PATCH_SIZE,
    PATCH_COUNT,
    PATCH_ERROR_SUFFIX,
    PATCH_METRIC_NAMES,
    deep_update,
    display_path,
    parse_csv_items,
    resolve_path,
    validate_patch_metric_array,
)
from vericodec_diff.sparsity_stats import concentration_at_percent, gini_coefficient
from vericodec_diff.verifier_features import DEFAULT_SIGNAL_NAMES, SIGNAL_SUFFIX


VERIFIER_CHECKPOINT_VERSION = 1
VERIFIER_EVAL_VERSION = 1
KILL_MEMO_VERSION = 2
CHECKPOINT_INDEX_FILENAME = "checkpoint_index__patch64.json"
TRAINING_SUMMARY_FILENAME = "training_summary__patch64.json"
EVAL_JSON_FILENAME = "verifier_eval__patch64.json"
BUDGET_CURVES_FILENAME = "verifier_budget_curves.csv"
DEFAULT_CHECKPOINT_DIR = "checkpoints/verifier"
DEFAULT_FEATURE_ROOT = "data/processed/verifier_features"
DEFAULT_ERROR_MAP_ROOT = "outputs/error_maps"
DEFAULT_EVAL_OUTPUT_DIR = "outputs/metrics/kill_test"
DEFAULT_MEMO_DIR = "outputs/memos"
DEFAULT_MEMO_MD_PATH = "outputs/memos/kill_test_memo.md"
DEFAULT_MEMO_JSON_PATH = "outputs/memos/kill_test_memo.json"
SUPPORTED_MODEL_TYPES = ("logistic", "mlp")
SUPPORTED_BUDGET_METRICS = ("positive_recall", "metric_mass_recovery")
LOCKED_KILL_GATE_PERCENT = 15
GENERATION_PROXY_BUDGET_METRIC = "metric_mass_recovery"
ZERO_DENOMINATOR_RATIO_SENTINEL = 1.0e12


class VerifierDatasetError(ValueError):
    """Raised when verifier features and patch-error labels cannot be aligned deterministically."""


class VerifierTrainingError(RuntimeError):
    """Raised when verifier model training or evaluation cannot proceed safely."""


@dataclass
class VerifierPatchSample:
    sample_id: str
    split: str
    feature_path: Path
    error_map_path: Path
    feature_matrix: np.ndarray
    label_metric_values: np.ndarray
    labels: np.ndarray


@dataclass
class VerifierDataset:
    name: str
    raw_splits: tuple[str, ...]
    signal_names: tuple[str, ...]
    label_metric_name: str
    label_threshold: float
    samples: list[VerifierPatchSample]
    features: np.ndarray
    labels: np.ndarray
    label_metric_values: np.ndarray
    sample_slices: list[tuple[int, int]]

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def patch_count(self) -> int:
        return int(self.labels.size)

    @property
    def positive_patch_count(self) -> int:
        return int(self.labels.sum(dtype=np.int64))

    @property
    def negative_patch_count(self) -> int:
        return int(self.labels.size - self.labels.sum(dtype=np.int64))


def _scalar_string(value: Any) -> str:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return str(value.item())
        raise ValueError("Expected a scalar string payload")
    return str(value)


def _scalar_int(value: Any) -> int:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return int(value.item())
        raise ValueError("Expected a scalar integer payload")
    return int(value)


def _scalar_float(value: Any) -> float:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return float(value.item())
        raise ValueError("Expected a scalar float payload")
    return float(value)


def _round_float(value: float) -> float:
    return float(round(float(value), 8))


def _validate_repo_relative_path(repo_root: Path, raw_path: str, *, name: str) -> Path:
    resolved = resolve_path(repo_root, raw_path).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"{name} must resolve inside repo_root for stable local paths") from exc
    return resolved


def _parse_signal_names(raw_value: str | Sequence[str] | None) -> list[str]:
    signal_names = parse_csv_items(raw_value)
    if not signal_names:
        signal_names = list(DEFAULT_SIGNAL_NAMES)
    invalid = [name for name in signal_names if not name]
    if invalid:
        raise ValueError("data.signal_names contains empty items")
    deduped: list[str] = []
    for signal_name in signal_names:
        if signal_name not in deduped:
            deduped.append(signal_name)
    return deduped


def _parse_split_list(raw_value: str | Sequence[str] | None, *, label: str) -> list[str]:
    split_names = parse_csv_items(raw_value)
    if not split_names:
        raise ValueError(f"{label} must contain at least one split name")
    return split_names


def _parse_budget_percents(raw_value: str | Sequence[int] | Sequence[str] | None) -> list[int]:
    if raw_value is None:
        values = [1, 2, 5, 10, 15, 20]
    elif isinstance(raw_value, str):
        values = [int(item) for item in parse_csv_items(raw_value)]
    else:
        values = [int(item) for item in raw_value]
    if not values:
        raise ValueError("evaluation.budget_percents must contain at least one percent")
    deduped: list[int] = []
    for value in values:
        if value <= 0 or value > 100:
            raise ValueError("evaluation.budget_percents must lie within [1, 100]")
        if value not in deduped:
            deduped.append(value)
    return deduped


def _parse_optional_probability_threshold(raw_value: Any, *, name: str) -> float | None:
    if raw_value in (None, ""):
        return None
    value = float(raw_value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must lie within [0, 1]")
    return value


def _parse_optional_nonnegative_threshold(raw_value: Any, *, name: str) -> float | None:
    if raw_value in (None, ""):
        return None
    value = float(raw_value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


def default_verifier_train_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "feature_root": DEFAULT_FEATURE_ROOT,
            "error_map_root": DEFAULT_ERROR_MAP_ROOT,
            "checkpoint_dir": DEFAULT_CHECKPOINT_DIR,
        },
        "data": {
            "signal_names": list(DEFAULT_SIGNAL_NAMES),
            "train_splits": ["train"],
            "val_splits": ["val"],
            "limit": None,
        },
        "labels": {
            "metric_name": "lpips",
            "threshold": 0.05,
        },
        "models": {
            "enabled": list(SUPPORTED_MODEL_TYPES),
            "seed": 0,
            "logistic": {
                "learning_rate": 0.1,
                "epochs": 400,
                "l2": 1e-4,
            },
            "mlp": {
                "hidden_dim": 16,
                "learning_rate": 0.01,
                "epochs": 300,
                "l2": 1e-4,
            },
        },
        "runtime": {
            "overwrite": True,
        },
    }


def default_verifier_eval_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "feature_root": DEFAULT_FEATURE_ROOT,
            "error_map_root": DEFAULT_ERROR_MAP_ROOT,
            "checkpoint_dir": DEFAULT_CHECKPOINT_DIR,
            "checkpoint_index": None,
            "output_dir": DEFAULT_EVAL_OUTPUT_DIR,
        },
        "data": {
            "signal_names": list(DEFAULT_SIGNAL_NAMES),
            "val_splits": ["val"],
            "test_splits": ["kill"],
            "limit": None,
        },
        "labels": {
            "metric_name": "lpips",
            "threshold": 0.05,
        },
        "evaluation": {
            "budget_percents": [1, 2, 5, 10, 15, 20],
        },
    }


def default_kill_memo_config() -> dict[str, Any]:
    return {
        "paths": {
            "repo_root": ".",
            "eval_json": str(Path(DEFAULT_EVAL_OUTPUT_DIR) / EVAL_JSON_FILENAME),
            "budget_curves_csv": str(Path(DEFAULT_EVAL_OUTPUT_DIR) / BUDGET_CURVES_FILENAME),
            "error_map_root": DEFAULT_ERROR_MAP_ROOT,
            "output_dir": DEFAULT_MEMO_DIR,
            "memo_md_path": DEFAULT_MEMO_MD_PATH,
            "memo_json_path": DEFAULT_MEMO_JSON_PATH,
        },
        "thresholds": {
            "concentration_at_15_min": None,
            "gini_mean_min": None,
            "verifier_auprc_min": None,
            "auprc_multiplier_over_best_heuristic_min": None,
            "generation_proxy_concentration_at_15_min": None,
        },
    }


def resolve_verifier_train_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_verifier_train_config(), raw_config)
    paths = dict(config.get("paths", {}))
    data = dict(config.get("data", {}))
    labels = dict(config.get("labels", {}))
    models = dict(config.get("models", {}))
    runtime = dict(config.get("runtime", {}))

    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()
    feature_root = _validate_repo_relative_path(
        resolved_repo_root,
        str(paths.get("feature_root", DEFAULT_FEATURE_ROOT)),
        name="paths.feature_root",
    )
    error_map_root = _validate_repo_relative_path(
        resolved_repo_root,
        str(paths.get("error_map_root", DEFAULT_ERROR_MAP_ROOT)),
        name="paths.error_map_root",
    )
    checkpoint_dir = _validate_repo_relative_path(
        resolved_repo_root,
        str(paths.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)),
        name="paths.checkpoint_dir",
    )

    signal_names = _parse_signal_names(data.get("signal_names"))
    train_splits = _parse_split_list(data.get("train_splits"), label="data.train_splits")
    val_splits = _parse_split_list(data.get("val_splits"), label="data.val_splits")

    limit_value = data.get("limit")
    limit = None if limit_value in (None, "") else int(limit_value)
    if limit is not None and limit <= 0:
        raise ValueError("data.limit must be a positive integer when provided")

    label_metric_name = str(labels.get("metric_name", "lpips")).strip()
    if label_metric_name not in PATCH_METRIC_NAMES:
        raise ValueError(f"labels.metric_name must be one of {PATCH_METRIC_NAMES}")
    label_threshold = float(labels.get("threshold", 0.05))
    if label_threshold < 0.0:
        raise ValueError("labels.threshold must be non-negative")

    enabled_models = parse_csv_items(models.get("enabled"))
    if not enabled_models:
        enabled_models = list(SUPPORTED_MODEL_TYPES)
    unsupported_models = [name for name in enabled_models if name not in SUPPORTED_MODEL_TYPES]
    if unsupported_models:
        raise ValueError(f"Unsupported models.enabled entries: {unsupported_models}")

    logistic = dict(models.get("logistic", {}))
    mlp = dict(models.get("mlp", {}))
    for model_name, settings in (("logistic", logistic), ("mlp", mlp)):
        learning_rate = float(settings.get("learning_rate", 0.01))
        epochs = int(settings.get("epochs", 1))
        l2 = float(settings.get("l2", 0.0))
        if learning_rate <= 0.0:
            raise ValueError(f"models.{model_name}.learning_rate must be positive")
        if epochs <= 0:
            raise ValueError(f"models.{model_name}.epochs must be positive")
        if l2 < 0.0:
            raise ValueError(f"models.{model_name}.l2 must be non-negative")
    hidden_dim = int(mlp.get("hidden_dim", 16))
    if hidden_dim <= 0:
        raise ValueError("models.mlp.hidden_dim must be positive")

    return {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "feature_root": str(feature_root),
            "error_map_root": str(error_map_root),
            "checkpoint_dir": str(checkpoint_dir),
        },
        "data": {
            "signal_names": signal_names,
            "train_splits": train_splits,
            "val_splits": val_splits,
            "limit": limit,
        },
        "labels": {
            "metric_name": label_metric_name,
            "threshold": label_threshold,
        },
        "models": {
            "enabled": enabled_models,
            "seed": int(models.get("seed", 0)),
            "logistic": {
                "learning_rate": float(logistic.get("learning_rate", 0.1)),
                "epochs": int(logistic.get("epochs", 400)),
                "l2": float(logistic.get("l2", 1e-4)),
            },
            "mlp": {
                "hidden_dim": hidden_dim,
                "learning_rate": float(mlp.get("learning_rate", 0.01)),
                "epochs": int(mlp.get("epochs", 300)),
                "l2": float(mlp.get("l2", 1e-4)),
            },
        },
        "runtime": {
            "overwrite": bool(runtime.get("overwrite", True)),
        },
    }


def resolve_verifier_eval_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_verifier_eval_config(), raw_config)
    paths = dict(config.get("paths", {}))
    data = dict(config.get("data", {}))
    labels = dict(config.get("labels", {}))
    evaluation = dict(config.get("evaluation", {}))

    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()
    feature_root = _validate_repo_relative_path(
        resolved_repo_root,
        str(paths.get("feature_root", DEFAULT_FEATURE_ROOT)),
        name="paths.feature_root",
    )
    error_map_root = _validate_repo_relative_path(
        resolved_repo_root,
        str(paths.get("error_map_root", DEFAULT_ERROR_MAP_ROOT)),
        name="paths.error_map_root",
    )
    checkpoint_dir = _validate_repo_relative_path(
        resolved_repo_root,
        str(paths.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)),
        name="paths.checkpoint_dir",
    )
    checkpoint_index_raw = paths.get("checkpoint_index")
    if checkpoint_index_raw in (None, ""):
        checkpoint_index = (checkpoint_dir / CHECKPOINT_INDEX_FILENAME).resolve()
    else:
        checkpoint_index = _validate_repo_relative_path(
            resolved_repo_root,
            str(checkpoint_index_raw),
            name="paths.checkpoint_index",
        )
    output_dir = _validate_repo_relative_path(
        resolved_repo_root,
        str(paths.get("output_dir", DEFAULT_EVAL_OUTPUT_DIR)),
        name="paths.output_dir",
    )

    signal_names = _parse_signal_names(data.get("signal_names"))
    val_splits = _parse_split_list(data.get("val_splits"), label="data.val_splits")
    test_splits = _parse_split_list(data.get("test_splits"), label="data.test_splits")
    limit_value = data.get("limit")
    limit = None if limit_value in (None, "") else int(limit_value)
    if limit is not None and limit <= 0:
        raise ValueError("data.limit must be a positive integer when provided")

    label_metric_name = str(labels.get("metric_name", "lpips")).strip()
    if label_metric_name not in PATCH_METRIC_NAMES:
        raise ValueError(f"labels.metric_name must be one of {PATCH_METRIC_NAMES}")
    label_threshold = float(labels.get("threshold", 0.05))
    if label_threshold < 0.0:
        raise ValueError("labels.threshold must be non-negative")

    budget_percents = _parse_budget_percents(evaluation.get("budget_percents"))

    return {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "feature_root": str(feature_root),
            "error_map_root": str(error_map_root),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_index": str(checkpoint_index),
            "output_dir": str(output_dir),
        },
        "data": {
            "signal_names": signal_names,
            "val_splits": val_splits,
            "test_splits": test_splits,
            "limit": limit,
        },
        "labels": {
            "metric_name": label_metric_name,
            "threshold": label_threshold,
        },
        "evaluation": {
            "budget_percents": budget_percents,
        },
    }


def resolve_kill_memo_config(repo_root: Path, raw_config: Mapping[str, Any]) -> dict[str, Any]:
    config = deep_update(default_kill_memo_config(), raw_config)
    paths = dict(config.get("paths", {}))
    thresholds = dict(config.get("thresholds", {}))

    resolved_repo_root = resolve_path(repo_root, str(paths.get("repo_root", "."))).resolve()
    eval_json = _validate_repo_relative_path(
        resolved_repo_root,
        str(paths.get("eval_json", str(Path(DEFAULT_EVAL_OUTPUT_DIR) / EVAL_JSON_FILENAME))),
        name="paths.eval_json",
    )
    budget_curves_csv = _validate_repo_relative_path(
        resolved_repo_root,
        str(paths.get("budget_curves_csv", str(Path(DEFAULT_EVAL_OUTPUT_DIR) / BUDGET_CURVES_FILENAME))),
        name="paths.budget_curves_csv",
    )
    error_map_root = _validate_repo_relative_path(
        resolved_repo_root,
        str(paths.get("error_map_root", DEFAULT_ERROR_MAP_ROOT)),
        name="paths.error_map_root",
    )
    output_dir = _validate_repo_relative_path(
        resolved_repo_root,
        str(paths.get("output_dir", DEFAULT_MEMO_DIR)),
        name="paths.output_dir",
    )
    memo_md_path = _validate_repo_relative_path(
        resolved_repo_root,
        str(paths.get("memo_md_path", DEFAULT_MEMO_MD_PATH)),
        name="paths.memo_md_path",
    )
    memo_json_path = _validate_repo_relative_path(
        resolved_repo_root,
        str(paths.get("memo_json_path", DEFAULT_MEMO_JSON_PATH)),
        name="paths.memo_json_path",
    )

    return {
        "paths": {
            "repo_root": str(resolved_repo_root),
            "eval_json": str(eval_json),
            "budget_curves_csv": str(budget_curves_csv),
            "error_map_root": str(error_map_root),
            "output_dir": str(output_dir),
            "memo_md_path": str(memo_md_path),
            "memo_json_path": str(memo_json_path),
        },
        "thresholds": {
            "concentration_at_15_min": _parse_optional_probability_threshold(
                thresholds.get("concentration_at_15_min"),
                name="thresholds.concentration_at_15_min",
            ),
            "gini_mean_min": _parse_optional_probability_threshold(
                thresholds.get("gini_mean_min"),
                name="thresholds.gini_mean_min",
            ),
            "verifier_auprc_min": _parse_optional_probability_threshold(
                thresholds.get("verifier_auprc_min"),
                name="thresholds.verifier_auprc_min",
            ),
            "auprc_multiplier_over_best_heuristic_min": _parse_optional_nonnegative_threshold(
                thresholds.get("auprc_multiplier_over_best_heuristic_min"),
                name="thresholds.auprc_multiplier_over_best_heuristic_min",
            ),
            "generation_proxy_concentration_at_15_min": _parse_optional_probability_threshold(
                thresholds.get("generation_proxy_concentration_at_15_min"),
                name="thresholds.generation_proxy_concentration_at_15_min",
            ),
        },
    }


def _discover_artifact_map(
    root: Path,
    *,
    suffix: str,
    selected_splits: Sequence[str],
    artifact_label: str,
) -> dict[tuple[str, str], Path]:
    if not root.exists():
        raise FileNotFoundError(f"{artifact_label} root not found: {root}")

    all_paths = sorted(root.rglob(f"*{suffix}"))
    if not all_paths:
        raise FileNotFoundError(f"No {artifact_label} files ending with {suffix} were found under {root}")

    available_splits = sorted({path.parent.name for path in all_paths})
    missing_splits = sorted(set(selected_splits) - set(available_splits))
    if missing_splits:
        raise VerifierDatasetError(
            f"{artifact_label} splits {missing_splits} were not found under {root}; available splits: {available_splits}"
        )

    output: dict[tuple[str, str], Path] = {}
    for path in all_paths:
        split = path.parent.name
        if split not in set(selected_splits):
            continue
        sample_id = path.name[: -len(suffix)]
        key = (split, sample_id)
        if key in output:
            raise VerifierDatasetError(f"Found duplicate {artifact_label} artifact for {split}/{sample_id}: {path}")
        output[key] = path.resolve()

    if not output:
        raise VerifierDatasetError(
            f"No {artifact_label} files matched splits {list(selected_splits)} under {root}"
        )
    return output


def _load_feature_matrix(path: Path, *, signal_names: Sequence[str]) -> tuple[str, str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        patch_size = _scalar_int(payload["patch_size"])
        grid_height = _scalar_int(payload["grid_height"])
        grid_width = _scalar_int(payload["grid_width"])
        if patch_size != DEFAULT_PATCH_SIZE:
            raise VerifierDatasetError(f"{path}: expected patch_size {DEFAULT_PATCH_SIZE}, found {patch_size}")
        if (grid_height, grid_width) != (DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE):
            raise VerifierDatasetError(
                f"{path}: expected grid {(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)}, found {(grid_height, grid_width)}"
            )

        missing_signals = [signal_name for signal_name in signal_names if signal_name not in payload.files]
        if missing_signals:
            raise VerifierDatasetError(
                f"{path}: missing verifier signals {missing_signals}; re-run extract_verifier_signals.py with matching settings"
            )

        matrix = np.stack(
            [
                validate_patch_metric_array(
                    payload[signal_name],
                    metric_name=f"{path.name}:{signal_name}",
                )
                for signal_name in signal_names
            ],
            axis=1,
        ).astype(np.float32)
        sample_id = _scalar_string(payload["sample_id"])
        split = _scalar_string(payload["split"])
    return sample_id, split, matrix


def _load_label_metric_values(path: Path, *, metric_name: str) -> tuple[str, str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        patch_size = _scalar_int(payload["patch_size"])
        image_size = _scalar_int(payload["image_size"])
        grid_height = _scalar_int(payload["grid_height"])
        grid_width = _scalar_int(payload["grid_width"])
        if patch_size != DEFAULT_PATCH_SIZE:
            raise VerifierDatasetError(f"{path}: expected patch_size {DEFAULT_PATCH_SIZE}, found {patch_size}")
        if image_size != DEFAULT_IMAGE_SIZE:
            raise VerifierDatasetError(f"{path}: expected image_size {DEFAULT_IMAGE_SIZE}, found {image_size}")
        if (grid_height, grid_width) != (DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE):
            raise VerifierDatasetError(
                f"{path}: expected grid {(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)}, found {(grid_height, grid_width)}"
            )
        if metric_name not in payload.files:
            raise VerifierDatasetError(
                f"{path}: missing patch metric {metric_name!r}; re-run compute_patch_errors.py with matching settings"
            )
        values = validate_patch_metric_array(payload[metric_name], metric_name=f"{path.name}:{metric_name}")
        sample_id = _scalar_string(payload["sample_id"])
        split = _scalar_string(payload["split"])
    return sample_id, split, values


def load_verifier_dataset(
    *,
    repo_root: Path,
    feature_root: Path,
    error_map_root: Path,
    signal_names: Sequence[str],
    label_metric_name: str,
    label_threshold: float,
    dataset_name: str,
    raw_splits: Sequence[str],
    limit: int | None = None,
) -> VerifierDataset:
    selected_splits = tuple(raw_splits)
    feature_map = _discover_artifact_map(
        feature_root,
        suffix=SIGNAL_SUFFIX,
        selected_splits=selected_splits,
        artifact_label="verifier feature",
    )
    error_map = _discover_artifact_map(
        error_map_root,
        suffix=PATCH_ERROR_SUFFIX,
        selected_splits=selected_splits,
        artifact_label="patch-error",
    )

    missing_error_maps = sorted(set(feature_map) - set(error_map))
    if missing_error_maps:
        preview = ", ".join(f"{split}/{sample_id}" for split, sample_id in missing_error_maps[:5])
        raise VerifierDatasetError(
            f"Missing patch-error labels for {len(missing_error_maps)} feature samples ({preview}); run compute_patch_errors.py first"
        )

    missing_features = sorted(set(error_map) - set(feature_map))
    if missing_features:
        preview = ", ".join(f"{split}/{sample_id}" for split, sample_id in missing_features[:5])
        raise VerifierDatasetError(
            f"Missing verifier features for {len(missing_features)} patch-error samples ({preview}); run extract_verifier_signals.py first"
        )

    matched_keys = sorted(feature_map)
    if limit is not None:
        matched_keys = matched_keys[:limit]
    if not matched_keys:
        raise VerifierDatasetError(
            f"No aligned verifier feature / patch-error samples matched {dataset_name} splits {list(selected_splits)}"
        )

    samples: list[VerifierPatchSample] = []
    flattened_features: list[np.ndarray] = []
    flattened_labels: list[np.ndarray] = []
    flattened_metric_values: list[np.ndarray] = []
    sample_slices: list[tuple[int, int]] = []
    offset = 0

    for split, sample_id in matched_keys:
        feature_path = feature_map[(split, sample_id)]
        error_map_path = error_map[(split, sample_id)]

        feature_sample_id, feature_split, feature_matrix = _load_feature_matrix(
            feature_path,
            signal_names=signal_names,
        )
        error_sample_id, error_split, label_metric_values = _load_label_metric_values(
            error_map_path,
            metric_name=label_metric_name,
        )
        if feature_sample_id != error_sample_id or feature_split != error_split:
            raise VerifierDatasetError(
                f"Feature / patch-error payload mismatch for {feature_path} and {error_map_path}"
            )
        labels = (label_metric_values >= float(label_threshold)).astype(np.int32)
        sample = VerifierPatchSample(
            sample_id=sample_id,
            split=split,
            feature_path=feature_path,
            error_map_path=error_map_path,
            feature_matrix=feature_matrix,
            label_metric_values=label_metric_values.astype(np.float32),
            labels=labels,
        )
        samples.append(sample)
        flattened_features.append(feature_matrix)
        flattened_labels.append(labels)
        flattened_metric_values.append(label_metric_values.astype(np.float32))
        sample_slices.append((offset, offset + PATCH_COUNT))
        offset += PATCH_COUNT

    dataset = VerifierDataset(
        name=dataset_name,
        raw_splits=selected_splits,
        signal_names=tuple(signal_names),
        label_metric_name=label_metric_name,
        label_threshold=float(label_threshold),
        samples=samples,
        features=np.concatenate(flattened_features, axis=0).astype(np.float32),
        labels=np.concatenate(flattened_labels, axis=0).astype(np.int32),
        label_metric_values=np.concatenate(flattened_metric_values, axis=0).astype(np.float32),
        sample_slices=sample_slices,
    )
    if dataset.positive_patch_count == 0 or dataset.negative_patch_count == 0:
        raise VerifierDatasetError(
            f"{dataset_name} dataset built from splits {list(selected_splits)} must contain both positive and negative patches"
        )
    return dataset


def dataset_summary(dataset: VerifierDataset) -> dict[str, Any]:
    positive_rate = float(dataset.positive_patch_count / max(1, dataset.patch_count))
    metric_mean = float(dataset.label_metric_values.mean(dtype=np.float64))
    return {
        "name": dataset.name,
        "raw_splits": list(dataset.raw_splits),
        "sample_count": dataset.sample_count,
        "patch_count": dataset.patch_count,
        "positive_patch_count": dataset.positive_patch_count,
        "negative_patch_count": dataset.negative_patch_count,
        "positive_patch_rate": _round_float(positive_rate),
        "label_metric_name": dataset.label_metric_name,
        "label_threshold": _round_float(dataset.label_threshold),
        "label_metric_mean": _round_float(metric_mean),
        "signal_names": list(dataset.signal_names),
    }


def _fit_standardization(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def _apply_standardization(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean) / std).astype(np.float32)


def _balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    positives = float(labels.sum(dtype=np.float64))
    negatives = float(labels.size - labels.sum(dtype=np.int64))
    if positives <= 0.0 or negatives <= 0.0:
        raise VerifierTrainingError("Balanced class weighting requires both positive and negative examples")
    weights = np.ones(labels.size, dtype=np.float32)
    weights[labels.astype(bool)] = np.float32(negatives / positives)
    weights *= np.float32(labels.size / max(1.0, float(weights.sum(dtype=np.float64))))
    return weights


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _weighted_logistic_loss(logits: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    targets = labels.astype(np.float32)
    losses = weights * (np.logaddexp(0.0, logits) - (targets * logits))
    return float(np.mean(losses, dtype=np.float64))


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int32).reshape(-1)
    y_score = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y_true.size != y_score.size:
        raise ValueError("labels and scores must have matching shapes")

    positives = int(y_true.sum(dtype=np.int64))
    negatives = int(y_true.size - positives)
    if positives <= 0 or negatives <= 0:
        raise VerifierTrainingError("AUPRC and AUROC require both positive and negative examples")

    order = np.argsort(-y_score, kind="mergesort")
    sorted_scores = y_score[order]
    sorted_labels = y_true[order]
    distinct_indices = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])

    tp = np.cumsum(sorted_labels, dtype=np.float64)[distinct_indices]
    fp = np.cumsum(1 - sorted_labels, dtype=np.float64)[distinct_indices]
    precision = tp / np.maximum(1.0, tp + fp)
    recall = tp / float(positives)
    auprc = float(np.sum((recall - np.r_[0.0, recall[:-1]]) * precision, dtype=np.float64))

    tpr = np.r_[0.0, recall, 1.0]
    fpr = np.r_[0.0, fp / float(negatives), 1.0]
    auroc = float(np.trapz(tpr, fpr))
    positive_rate = float(positives / y_true.size)
    return {
        "auprc": _round_float(auprc),
        "auroc": _round_float(auroc),
        "positive_patch_rate": _round_float(positive_rate),
    }


def _predict_logistic(features: np.ndarray, *, weights: np.ndarray, bias: float) -> np.ndarray:
    return _sigmoid((features @ weights) + np.float32(bias))


def _predict_mlp(
    features: np.ndarray,
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
) -> np.ndarray:
    hidden = np.tanh((features @ w1) + b1).astype(np.float32)
    logits = (hidden @ w2).reshape(-1) + np.float32(b2)
    return _sigmoid(logits)


def _train_logistic(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    *,
    learning_rate: float,
    epochs: int,
    l2: float,
) -> dict[str, Any]:
    weights = np.zeros(train_features.shape[1], dtype=np.float32)
    bias = np.float32(0.0)
    sample_weights = _balanced_sample_weights(train_labels)
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        logits = (train_features @ weights) + bias
        probabilities = _sigmoid(logits)
        gradient_scale = (sample_weights * (probabilities - train_labels.astype(np.float32))) / np.float32(
            train_labels.size
        )
        grad_w = (train_features.T @ gradient_scale) + (np.float32(l2) * weights)
        grad_b = np.sum(gradient_scale, dtype=np.float64).astype(np.float32)
        weights -= np.float32(learning_rate) * grad_w
        bias -= np.float32(learning_rate) * grad_b

        val_scores = _predict_logistic(val_features, weights=weights, bias=float(bias))
        val_metrics = _binary_metrics(val_labels, val_scores)
        train_loss = _weighted_logistic_loss(logits, train_labels, sample_weights) + (
            0.5 * float(l2) * float(np.sum(np.square(weights), dtype=np.float64))
        )
        is_better = False
        if best_metrics is None:
            is_better = True
        elif val_metrics["auprc"] > best_metrics["auprc"]:
            is_better = True
        elif val_metrics["auprc"] == best_metrics["auprc"] and val_metrics["auroc"] > best_metrics["auroc"]:
            is_better = True
        if is_better:
            best_state = {
                "weights": weights.copy(),
                "bias": np.asarray(float(bias), dtype=np.float32),
            }
            best_metrics = dict(val_metrics)
            best_epoch = epoch
            best_loss = float(train_loss)

    if best_state is None or best_metrics is None:
        raise VerifierTrainingError("Failed to retain a best logistic checkpoint state")
    return {
        "state": best_state,
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "best_train_loss": _round_float(best_loss),
    }


def _train_mlp(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    *,
    hidden_dim: int,
    learning_rate: float,
    epochs: int,
    l2: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    input_dim = train_features.shape[1]
    w1 = rng.normal(0.0, np.sqrt(1.0 / max(1, input_dim)), size=(input_dim, hidden_dim)).astype(np.float32)
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    w2 = rng.normal(0.0, np.sqrt(1.0 / max(1, hidden_dim)), size=(hidden_dim, 1)).astype(np.float32)
    b2 = np.float32(0.0)

    sample_weights = _balanced_sample_weights(train_labels)
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    best_loss = float("inf")

    train_targets = train_labels.astype(np.float32)
    for epoch in range(1, epochs + 1):
        hidden = np.tanh((train_features @ w1) + b1).astype(np.float32)
        logits = (hidden @ w2).reshape(-1) + b2
        probabilities = _sigmoid(logits)

        gradient_logits = (sample_weights * (probabilities - train_targets)) / np.float32(train_labels.size)
        grad_w2 = (hidden.T @ gradient_logits[:, None]) + (np.float32(l2) * w2)
        grad_b2 = np.sum(gradient_logits, dtype=np.float64).astype(np.float32)
        hidden_gradient = (gradient_logits[:, None] @ w2.T) * (1.0 - np.square(hidden, dtype=np.float32))
        grad_w1 = (train_features.T @ hidden_gradient) + (np.float32(l2) * w1)
        grad_b1 = np.sum(hidden_gradient, axis=0, dtype=np.float64).astype(np.float32)

        w2 -= np.float32(learning_rate) * grad_w2
        b2 -= np.float32(learning_rate) * grad_b2
        w1 -= np.float32(learning_rate) * grad_w1
        b1 -= np.float32(learning_rate) * grad_b1

        val_scores = _predict_mlp(val_features, w1=w1, b1=b1, w2=w2, b2=float(b2))
        val_metrics = _binary_metrics(val_labels, val_scores)
        train_loss = _weighted_logistic_loss(logits, train_labels, sample_weights)
        train_loss += 0.5 * float(l2) * float(
            np.sum(np.square(w1), dtype=np.float64) + np.sum(np.square(w2), dtype=np.float64)
        )

        is_better = False
        if best_metrics is None:
            is_better = True
        elif val_metrics["auprc"] > best_metrics["auprc"]:
            is_better = True
        elif val_metrics["auprc"] == best_metrics["auprc"] and val_metrics["auroc"] > best_metrics["auroc"]:
            is_better = True
        if is_better:
            best_state = {
                "w1": w1.copy(),
                "b1": b1.copy(),
                "w2": w2.copy(),
                "b2": np.asarray(float(b2), dtype=np.float32),
            }
            best_metrics = dict(val_metrics)
            best_epoch = epoch
            best_loss = float(train_loss)

    if best_state is None or best_metrics is None:
        raise VerifierTrainingError("Failed to retain a best MLP checkpoint state")
    return {
        "state": best_state,
        "best_epoch": best_epoch,
        "best_val_metrics": best_metrics,
        "best_train_loss": _round_float(best_loss),
    }


def _checkpoint_filename(model_type: str, *, label_metric_name: str, label_threshold: float, hidden_dim: int | None) -> str:
    threshold_text = f"{label_threshold:.4f}".replace(".", "p")
    if model_type == "logistic":
        return f"verifier__logistic__patch64__{label_metric_name}_ge_{threshold_text}.npz"
    if model_type == "mlp":
        return f"verifier__mlp_h{int(hidden_dim or 0)}__patch64__{label_metric_name}_ge_{threshold_text}.npz"
    raise ValueError(f"Unsupported model_type {model_type!r}")


def _predict_from_checkpoint_payload(checkpoint: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    normalization_mean = np.asarray(checkpoint["normalization_mean"], dtype=np.float32)
    normalization_std = np.asarray(checkpoint["normalization_std"], dtype=np.float32)
    normalized = _apply_standardization(features.astype(np.float32), normalization_mean, normalization_std)
    model_type = str(checkpoint["model_type"])
    if model_type == "logistic":
        return _predict_logistic(
            normalized,
            weights=np.asarray(checkpoint["weights"], dtype=np.float32),
            bias=float(_scalar_float(checkpoint["bias"])),
        )
    if model_type == "mlp":
        return _predict_mlp(
            normalized,
            w1=np.asarray(checkpoint["w1"], dtype=np.float32),
            b1=np.asarray(checkpoint["b1"], dtype=np.float32),
            w2=np.asarray(checkpoint["w2"], dtype=np.float32),
            b2=float(_scalar_float(checkpoint["b2"])),
        )
    raise ValueError(f"Unsupported checkpoint model_type {model_type!r}")


def train_verifier_models(config: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    feature_root = Path(config["paths"]["feature_root"])
    error_map_root = Path(config["paths"]["error_map_root"])
    checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_verifier_dataset(
        repo_root=repo_root,
        feature_root=feature_root,
        error_map_root=error_map_root,
        signal_names=config["data"]["signal_names"],
        label_metric_name=str(config["labels"]["metric_name"]),
        label_threshold=float(config["labels"]["threshold"]),
        dataset_name="train",
        raw_splits=config["data"]["train_splits"],
        limit=config["data"]["limit"],
    )
    val_dataset = load_verifier_dataset(
        repo_root=repo_root,
        feature_root=feature_root,
        error_map_root=error_map_root,
        signal_names=config["data"]["signal_names"],
        label_metric_name=str(config["labels"]["metric_name"]),
        label_threshold=float(config["labels"]["threshold"]),
        dataset_name="val",
        raw_splits=config["data"]["val_splits"],
        limit=config["data"]["limit"],
    )

    normalization_mean, normalization_std = _fit_standardization(train_dataset.features)
    train_features = _apply_standardization(train_dataset.features, normalization_mean, normalization_std)
    val_features = _apply_standardization(val_dataset.features, normalization_mean, normalization_std)

    checkpoint_records: list[dict[str, Any]] = []
    enabled_models = list(config["models"]["enabled"])
    for model_type in enabled_models:
        if model_type == "logistic":
            training_result = _train_logistic(
                train_features,
                train_dataset.labels,
                val_features,
                val_dataset.labels,
                learning_rate=float(config["models"]["logistic"]["learning_rate"]),
                epochs=int(config["models"]["logistic"]["epochs"]),
                l2=float(config["models"]["logistic"]["l2"]),
            )
            state = dict(training_result["state"])
            hidden_dim = None
        elif model_type == "mlp":
            training_result = _train_mlp(
                train_features,
                train_dataset.labels,
                val_features,
                val_dataset.labels,
                hidden_dim=int(config["models"]["mlp"]["hidden_dim"]),
                learning_rate=float(config["models"]["mlp"]["learning_rate"]),
                epochs=int(config["models"]["mlp"]["epochs"]),
                l2=float(config["models"]["mlp"]["l2"]),
                seed=int(config["models"]["seed"]),
            )
            state = dict(training_result["state"])
            hidden_dim = int(config["models"]["mlp"]["hidden_dim"])
        else:
            raise ValueError(f"Unsupported model_type {model_type!r}")

        checkpoint_filename = _checkpoint_filename(
            model_type,
            label_metric_name=str(config["labels"]["metric_name"]),
            label_threshold=float(config["labels"]["threshold"]),
            hidden_dim=hidden_dim,
        )
        checkpoint_path = checkpoint_dir / checkpoint_filename
        checkpoint_payload: dict[str, Any] = {
            "version": np.asarray(VERIFIER_CHECKPOINT_VERSION, dtype=np.int32),
            "model_type": np.asarray(model_type),
            "signal_names": np.asarray(list(config["data"]["signal_names"])),
            "label_metric_name": np.asarray(str(config["labels"]["metric_name"])),
            "label_threshold": np.asarray(float(config["labels"]["threshold"]), dtype=np.float32),
            "patch_size": np.asarray(DEFAULT_PATCH_SIZE, dtype=np.int32),
            "grid_height": np.asarray(DEFAULT_GRID_SIZE, dtype=np.int32),
            "grid_width": np.asarray(DEFAULT_GRID_SIZE, dtype=np.int32),
            "normalization_mean": normalization_mean.astype(np.float32),
            "normalization_std": normalization_std.astype(np.float32),
            "seed": np.asarray(int(config["models"]["seed"]), dtype=np.int32),
            "hidden_dim": np.asarray(hidden_dim or 0, dtype=np.int32),
            "best_epoch": np.asarray(int(training_result["best_epoch"]), dtype=np.int32),
            "best_train_loss": np.asarray(float(training_result["best_train_loss"]), dtype=np.float32),
            "best_val_auprc": np.asarray(float(training_result["best_val_metrics"]["auprc"]), dtype=np.float32),
            "best_val_auroc": np.asarray(float(training_result["best_val_metrics"]["auroc"]), dtype=np.float32),
        }
        checkpoint_payload.update(state)
        np.savez_compressed(checkpoint_path, **checkpoint_payload)

        train_scores = _predict_from_checkpoint_payload(checkpoint_payload, train_dataset.features)
        val_scores = _predict_from_checkpoint_payload(checkpoint_payload, val_dataset.features)
        train_metrics = _binary_metrics(train_dataset.labels, train_scores)
        val_metrics = _binary_metrics(val_dataset.labels, val_scores)
        checkpoint_records.append(
            {
                "model_type": model_type,
                "checkpoint_path": display_path(checkpoint_path, repo_root),
                "checkpoint_filename": checkpoint_filename,
                "hidden_dim": hidden_dim,
                "best_epoch": int(training_result["best_epoch"]),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "training_config": (
                    dict(config["models"][model_type])
                    if model_type in config["models"]
                    else {}
                ),
            }
        )

    checkpoint_records = sorted(
        checkpoint_records,
        key=lambda item: (
            -float(item["val_metrics"]["auprc"]),
            -float(item["val_metrics"]["auroc"]),
            str(item["checkpoint_filename"]),
        ),
    )
    best_checkpoint = checkpoint_records[0]

    index_path = checkpoint_dir / CHECKPOINT_INDEX_FILENAME
    training_summary_path = checkpoint_dir / TRAINING_SUMMARY_FILENAME
    index_payload = {
        "version": VERIFIER_CHECKPOINT_VERSION,
        "signal_names": list(config["data"]["signal_names"]),
        "label_definition": {
            "metric_name": str(config["labels"]["metric_name"]),
            "threshold": _round_float(float(config["labels"]["threshold"])),
        },
        "checkpoints": checkpoint_records,
        "best_checkpoint": {
            "checkpoint_filename": best_checkpoint["checkpoint_filename"],
            "checkpoint_path": best_checkpoint["checkpoint_path"],
            "model_type": best_checkpoint["model_type"],
            "val_auprc": best_checkpoint["val_metrics"]["auprc"],
            "val_auroc": best_checkpoint["val_metrics"]["auroc"],
        },
    }
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(index_payload, handle, indent=2, sort_keys=True)

    training_summary = {
        "version": VERIFIER_CHECKPOINT_VERSION,
        "train_dataset": dataset_summary(train_dataset),
        "val_dataset": dataset_summary(val_dataset),
        "signal_names": list(config["data"]["signal_names"]),
        "label_definition": {
            "metric_name": str(config["labels"]["metric_name"]),
            "threshold": _round_float(float(config["labels"]["threshold"])),
        },
        "checkpoint_index": display_path(index_path, repo_root),
        "best_checkpoint": index_payload["best_checkpoint"],
        "checkpoints": checkpoint_records,
    }
    with training_summary_path.open("w", encoding="utf-8") as handle:
        json.dump(training_summary, handle, indent=2, sort_keys=True)

    return {
        "train_dataset": dataset_summary(train_dataset),
        "val_dataset": dataset_summary(val_dataset),
        "checkpoint_index_path": index_path,
        "training_summary_path": training_summary_path,
        "checkpoints": checkpoint_records,
        "best_checkpoint": best_checkpoint,
    }


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        output = {name: payload[name] for name in payload.files}
    patch_size = _scalar_int(output["patch_size"])
    grid_height = _scalar_int(output["grid_height"])
    grid_width = _scalar_int(output["grid_width"])
    if patch_size != DEFAULT_PATCH_SIZE:
        raise VerifierTrainingError(f"{path}: expected patch_size {DEFAULT_PATCH_SIZE}, found {patch_size}")
    if (grid_height, grid_width) != (DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE):
        raise VerifierTrainingError(
            f"{path}: expected grid {(DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE)}, found {(grid_height, grid_width)}"
        )
    return output


def _validate_checkpoint_compatibility(
    checkpoint: Mapping[str, Any],
    *,
    signal_names: Sequence[str],
    label_metric_name: str,
    label_threshold: float,
    path: Path,
) -> None:
    checkpoint_signal_names = [str(item) for item in np.asarray(checkpoint["signal_names"]).tolist()]
    if checkpoint_signal_names != list(signal_names):
        raise VerifierTrainingError(
            f"{path}: checkpoint signal_names {checkpoint_signal_names} do not match requested {list(signal_names)}"
        )
    checkpoint_metric = _scalar_string(checkpoint["label_metric_name"])
    if checkpoint_metric != label_metric_name:
        raise VerifierTrainingError(
            f"{path}: checkpoint label metric {checkpoint_metric!r} does not match requested {label_metric_name!r}"
        )
    checkpoint_threshold = _scalar_float(checkpoint["label_threshold"])
    if abs(checkpoint_threshold - float(label_threshold)) > 1e-8:
        raise VerifierTrainingError(
            f"{path}: checkpoint label threshold {checkpoint_threshold} does not match requested {label_threshold}"
        )


def _best_record(records: Sequence[Mapping[str, Any]], *, key_prefix: str) -> Mapping[str, Any]:
    return sorted(
        records,
        key=lambda item: (
            -float(item[f"{key_prefix}_metrics"]["auprc"]),
            -float(item[f"{key_prefix}_metrics"]["auroc"]),
            str(item.get("checkpoint_filename", item.get("signal_name", ""))),
        ),
    )[0]


def select_best_heuristic_baseline(
    *,
    val_dataset: VerifierDataset,
    test_dataset: VerifierDataset,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for signal_index, signal_name in enumerate(val_dataset.signal_names):
        for direction in (1, -1):
            direction_name = "positive" if direction > 0 else "negative"
            val_scores = np.float32(direction) * val_dataset.features[:, signal_index]
            test_scores = np.float32(direction) * test_dataset.features[:, signal_index]
            candidates.append(
                {
                    "signal_name": signal_name,
                    "direction": int(direction),
                    "direction_name": direction_name,
                    "val_metrics": _binary_metrics(val_dataset.labels, val_scores),
                    "test_metrics": _binary_metrics(test_dataset.labels, test_scores),
                }
            )
    best_candidate = _best_record(candidates, key_prefix="val")
    return {
        "family": "best_single_signal",
        "signal_name": str(best_candidate["signal_name"]),
        "direction": int(best_candidate["direction"]),
        "direction_name": str(best_candidate["direction_name"]),
        "val_metrics": dict(best_candidate["val_metrics"]),
        "test_metrics": dict(best_candidate["test_metrics"]),
        "candidates": candidates,
    }


def _dataset_sample_scores(dataset: VerifierDataset, flat_scores: np.ndarray) -> list[np.ndarray]:
    if flat_scores.shape[0] != dataset.patch_count:
        raise ValueError("flat_scores must align with dataset.patch_count")
    outputs: list[np.ndarray] = []
    for start, stop in dataset.sample_slices:
        outputs.append(np.asarray(flat_scores[start:stop], dtype=np.float32))
    return outputs


def compute_budget_curve_rows(
    *,
    dataset: VerifierDataset,
    flat_scores: np.ndarray,
    budget_percents: Sequence[int],
    curve_role: str,
    model_name: str,
) -> list[dict[str, Any]]:
    sample_scores = _dataset_sample_scores(dataset, flat_scores)
    rows: list[dict[str, Any]] = []
    for budget_percent in budget_percents:
        selected_patch_count = max(1, int(np.ceil(PATCH_COUNT * (float(budget_percent) / 100.0))))
        positive_recall_values: list[float] = []
        metric_mass_values: list[float] = []
        precision_values: list[float] = []

        for sample, scores in zip(dataset.samples, sample_scores):
            order = np.argsort(-scores, kind="mergesort")[:selected_patch_count]
            labels = sample.labels.astype(np.int32)
            metric_values = sample.label_metric_values.astype(np.float32)
            positive_count = int(labels.sum(dtype=np.int64))
            if positive_count > 0:
                positive_recall_values.append(float(labels[order].sum(dtype=np.int64) / positive_count))
                precision_values.append(float(labels[order].mean(dtype=np.float64)))
            metric_total = float(metric_values.sum(dtype=np.float64))
            if metric_total > 0.0:
                metric_mass_values.append(float(metric_values[order].sum(dtype=np.float64) / metric_total))

        rows.append(
            {
                "curve_role": curve_role,
                "model_name": model_name,
                "dataset_name": dataset.name,
                "budget_percent": int(budget_percent),
                "selected_patch_count": int(selected_patch_count),
                "sample_count": int(dataset.sample_count),
                "samples_with_positives": int(len(positive_recall_values)),
                "positive_recall": _round_float(
                    float(np.mean(positive_recall_values, dtype=np.float64)) if positive_recall_values else 0.0
                ),
                "metric_mass_recovery": _round_float(
                    float(np.mean(metric_mass_values, dtype=np.float64)) if metric_mass_values else 0.0
                ),
                "precision_at_budget": _round_float(
                    float(np.mean(precision_values, dtype=np.float64)) if precision_values else 0.0
                ),
            }
        )
    return rows


def write_budget_curve_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "curve_role",
        "model_name",
        "dataset_name",
        "budget_percent",
        "selected_patch_count",
        "sample_count",
        "samples_with_positives",
        "positive_recall",
        "metric_mass_recovery",
        "precision_at_budget",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (str(item["curve_role"]), int(item["budget_percent"]))):
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def evaluate_verifier_models(config: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    feature_root = Path(config["paths"]["feature_root"])
    error_map_root = Path(config["paths"]["error_map_root"])
    checkpoint_index_path = Path(config["paths"]["checkpoint_index"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_index_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint index not found: {checkpoint_index_path}; run scripts/train_verifier.py first"
        )
    with checkpoint_index_path.open("r", encoding="utf-8") as handle:
        checkpoint_index = json.load(handle)
    checkpoint_records = list(checkpoint_index.get("checkpoints", []))
    if not checkpoint_records:
        raise VerifierTrainingError(f"{checkpoint_index_path} does not list any verifier checkpoints")

    val_dataset = load_verifier_dataset(
        repo_root=repo_root,
        feature_root=feature_root,
        error_map_root=error_map_root,
        signal_names=config["data"]["signal_names"],
        label_metric_name=str(config["labels"]["metric_name"]),
        label_threshold=float(config["labels"]["threshold"]),
        dataset_name="val",
        raw_splits=config["data"]["val_splits"],
        limit=config["data"]["limit"],
    )
    test_dataset = load_verifier_dataset(
        repo_root=repo_root,
        feature_root=feature_root,
        error_map_root=error_map_root,
        signal_names=config["data"]["signal_names"],
        label_metric_name=str(config["labels"]["metric_name"]),
        label_threshold=float(config["labels"]["threshold"]),
        dataset_name="test",
        raw_splits=config["data"]["test_splits"],
        limit=config["data"]["limit"],
    )

    model_evaluations: list[dict[str, Any]] = []
    best_checkpoint_payload: dict[str, Any] | None = None
    best_model_test_scores: np.ndarray | None = None

    for checkpoint_record in checkpoint_records:
        checkpoint_path = checkpoint_record.get("checkpoint_path")
        if not checkpoint_path:
            raise VerifierTrainingError(f"{checkpoint_index_path} contains a checkpoint record without checkpoint_path")
        resolved_checkpoint_path = resolve_path(repo_root, str(checkpoint_path)).resolve()
        checkpoint_payload = load_checkpoint_payload(resolved_checkpoint_path)
        _validate_checkpoint_compatibility(
            checkpoint_payload,
            signal_names=config["data"]["signal_names"],
            label_metric_name=str(config["labels"]["metric_name"]),
            label_threshold=float(config["labels"]["threshold"]),
            path=resolved_checkpoint_path,
        )

        val_scores = _predict_from_checkpoint_payload(checkpoint_payload, val_dataset.features)
        test_scores = _predict_from_checkpoint_payload(checkpoint_payload, test_dataset.features)
        evaluation_record = {
            "model_type": str(_scalar_string(checkpoint_payload["model_type"])),
            "checkpoint_filename": str(Path(resolved_checkpoint_path).name),
            "checkpoint_path": display_path(resolved_checkpoint_path, repo_root),
            "hidden_dim": int(_scalar_int(checkpoint_payload["hidden_dim"])),
            "best_epoch": int(_scalar_int(checkpoint_payload["best_epoch"])),
            "val_metrics": _binary_metrics(val_dataset.labels, val_scores),
            "test_metrics": _binary_metrics(test_dataset.labels, test_scores),
        }
        model_evaluations.append(evaluation_record)

    best_model = _best_record(model_evaluations, key_prefix="val")
    best_checkpoint_path = resolve_path(repo_root, str(best_model["checkpoint_path"])).resolve()
    best_checkpoint_payload = load_checkpoint_payload(best_checkpoint_path)
    best_model_test_scores = _predict_from_checkpoint_payload(best_checkpoint_payload, test_dataset.features)

    heuristic = select_best_heuristic_baseline(val_dataset=val_dataset, test_dataset=test_dataset)
    heuristic_test_scores = np.float32(heuristic["direction"]) * test_dataset.features[
        :, test_dataset.signal_names.index(str(heuristic["signal_name"]))
    ]

    budget_rows = compute_budget_curve_rows(
        dataset=test_dataset,
        flat_scores=best_model_test_scores,
        budget_percents=config["evaluation"]["budget_percents"],
        curve_role="best_model",
        model_name=str(best_model["checkpoint_filename"]),
    )
    budget_rows.extend(
        compute_budget_curve_rows(
            dataset=test_dataset,
            flat_scores=heuristic_test_scores,
            budget_percents=config["evaluation"]["budget_percents"],
            curve_role="heuristic_baseline",
            model_name=f"{heuristic['signal_name']}::{heuristic['direction_name']}",
        )
    )
    budget_curve_path = output_dir / BUDGET_CURVES_FILENAME
    write_budget_curve_csv(budget_curve_path, budget_rows)

    eval_json_path = output_dir / EVAL_JSON_FILENAME
    payload = {
        "version": VERIFIER_EVAL_VERSION,
        "patch_size": DEFAULT_PATCH_SIZE,
        "grid_size": [DEFAULT_GRID_SIZE, DEFAULT_GRID_SIZE],
        "signal_names": list(config["data"]["signal_names"]),
        "label_definition": {
            "metric_name": str(config["labels"]["metric_name"]),
            "threshold": _round_float(float(config["labels"]["threshold"])),
        },
        "datasets": {
            "val": dataset_summary(val_dataset),
            "test": dataset_summary(test_dataset),
        },
        "checkpoint_index": display_path(checkpoint_index_path, repo_root),
        "model_candidates": model_evaluations,
        "best_model": {
            **dict(best_model),
            "selection_metric": "val_auprc",
        },
        "heuristic_baseline": {
            "family": heuristic["family"],
            "signal_name": heuristic["signal_name"],
            "direction": heuristic["direction"],
            "direction_name": heuristic["direction_name"],
            "val_metrics": heuristic["val_metrics"],
            "test_metrics": heuristic["test_metrics"],
            "delta_vs_best_model": {
                "auprc": _round_float(
                    float(best_model["test_metrics"]["auprc"]) - float(heuristic["test_metrics"]["auprc"])
                ),
                "auroc": _round_float(
                    float(best_model["test_metrics"]["auroc"]) - float(heuristic["test_metrics"]["auroc"])
                ),
            },
        },
        "artifacts": {
            "budget_curves_csv": display_path(budget_curve_path, repo_root),
        },
    }
    with eval_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    return {
        "eval_json_path": eval_json_path,
        "budget_curve_path": budget_curve_path,
        "payload": payload,
        "budget_rows": budget_rows,
    }


def _find_budget_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    curve_role: str,
    budget_percent: int,
) -> Mapping[str, Any]:
    for row in rows:
        if str(row["curve_role"]) == curve_role and int(row["budget_percent"]) == int(budget_percent):
            return row
    raise VerifierTrainingError(
        f"Budget curve CSV does not contain curve_role={curve_role!r} at budget_percent={budget_percent}"
    )


def _render_bool(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _ratio_with_zero_guard(numerator: float, denominator: float) -> float:
    if denominator > 0.0:
        return float(numerator / denominator)
    if numerator <= 0.0:
        return 0.0
    return ZERO_DENOMINATOR_RATIO_SENTINEL


def _compute_failure_sparsity_summary(
    *,
    error_map_root: Path,
    selected_splits: Sequence[str],
    metric_name: str,
) -> dict[str, Any]:
    error_map = _discover_artifact_map(
        error_map_root,
        suffix=PATCH_ERROR_SUFFIX,
        selected_splits=selected_splits,
        artifact_label="patch-error",
    )
    concentration_values: list[float] = []
    gini_values: list[float] = []
    for split, sample_id in sorted(error_map):
        sample_path = error_map[(split, sample_id)]
        loaded_sample_id, loaded_split, metric_values = _load_label_metric_values(
            sample_path,
            metric_name=metric_name,
        )
        if loaded_sample_id != sample_id or loaded_split != split:
            raise VerifierTrainingError(f"Patch-error payload mismatch for {sample_path}")
        concentration_values.append(concentration_at_percent(metric_values, LOCKED_KILL_GATE_PERCENT))
        gini_values.append(gini_coefficient(metric_values))

    if not concentration_values:
        raise VerifierTrainingError(
            f"No patch-error artifacts were available to compute kill-gate sparsity for splits {list(selected_splits)}"
        )

    return {
        "metric_name": metric_name,
        "test_splits": [str(split) for split in selected_splits],
        "sample_count": int(len(concentration_values)),
        "mean_concentration_at_15": _round_float(float(np.mean(concentration_values, dtype=np.float64))),
        "mean_gini": _round_float(float(np.mean(gini_values, dtype=np.float64))),
    }


def write_kill_memo(config: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = Path(config["paths"]["repo_root"])
    eval_json_path = Path(config["paths"]["eval_json"])
    budget_curves_csv_path = Path(config["paths"]["budget_curves_csv"])
    error_map_root = Path(config["paths"]["error_map_root"])
    memo_md_path = Path(config["paths"]["memo_md_path"])
    memo_json_path = Path(config["paths"]["memo_json_path"])

    if not eval_json_path.is_file():
        raise FileNotFoundError(
            f"Verifier evaluation JSON not found: {eval_json_path}; run scripts/eval_verifier.py first"
        )
    if not budget_curves_csv_path.is_file():
        raise FileNotFoundError(
            f"Verifier budget curves CSV not found: {budget_curves_csv_path}; run scripts/eval_verifier.py first"
        )

    with eval_json_path.open("r", encoding="utf-8") as handle:
        evaluation = json.load(handle)
    with budget_curves_csv_path.open("r", encoding="utf-8", newline="") as handle:
        budget_rows = list(csv.DictReader(handle))

    thresholds = dict(config["thresholds"])
    configured_gate_count = 0
    checks: list[dict[str, Any]] = []

    best_model_metrics = dict(evaluation["best_model"]["test_metrics"])
    heuristic_metrics = dict(evaluation["heuristic_baseline"]["test_metrics"])
    label_definition = dict(evaluation["label_definition"])
    test_dataset_summary = dict(evaluation["datasets"]["test"])
    test_splits = [str(split) for split in test_dataset_summary.get("raw_splits", [])]
    if not test_splits:
        raise VerifierTrainingError("Evaluation JSON is missing datasets.test.raw_splits required for the kill memo")

    failure_sparsity = _compute_failure_sparsity_summary(
        error_map_root=error_map_root,
        selected_splits=test_splits,
        metric_name=str(label_definition["metric_name"]),
    )
    best_model_auprc = float(best_model_metrics["auprc"])
    heuristic_auprc = float(heuristic_metrics["auprc"])
    auprc_multiplier = _ratio_with_zero_guard(best_model_auprc, heuristic_auprc)
    generation_proxy_row = _find_budget_row(
        budget_rows,
        curve_role="best_model",
        budget_percent=LOCKED_KILL_GATE_PERCENT,
    )
    generation_proxy_concentration_at_15 = float(generation_proxy_row[GENERATION_PROXY_BUDGET_METRIC])
    measured_metrics = {
        "failure_concentration_at_15": failure_sparsity["mean_concentration_at_15"],
        "failure_gini_mean": failure_sparsity["mean_gini"],
        "verifier_test_auprc": _round_float(best_model_auprc),
        "verifier_test_auroc": _round_float(float(best_model_metrics["auroc"])),
        "heuristic_test_auprc": _round_float(heuristic_auprc),
        "heuristic_test_auroc": _round_float(float(heuristic_metrics["auroc"])),
        "auprc_multiplier_over_best_heuristic": _round_float(auprc_multiplier),
        "generation_proxy_concentration_at_15": _round_float(generation_proxy_concentration_at_15),
    }

    concentration_at_15_min = thresholds.get("concentration_at_15_min")
    if concentration_at_15_min is not None:
        configured_gate_count += 1
        measured = float(measured_metrics["failure_concentration_at_15"])
        threshold = float(concentration_at_15_min)
        checks.append(
            {
                "name": "failure_concentration_at_15",
                "measured": _round_float(measured),
                "threshold": _round_float(threshold),
                "comparison": ">=",
                "passed": measured >= threshold,
            }
        )

    gini_mean_min = thresholds.get("gini_mean_min")
    if gini_mean_min is not None:
        configured_gate_count += 1
        measured = float(measured_metrics["failure_gini_mean"])
        threshold = float(gini_mean_min)
        checks.append(
            {
                "name": "failure_gini_mean",
                "measured": _round_float(measured),
                "threshold": _round_float(threshold),
                "comparison": ">=",
                "passed": measured >= threshold,
            }
        )

    verifier_auprc_min = thresholds.get("verifier_auprc_min")
    if verifier_auprc_min is not None:
        configured_gate_count += 1
        measured = float(measured_metrics["verifier_test_auprc"])
        threshold = float(verifier_auprc_min)
        checks.append(
            {
                "name": "verifier_test_auprc",
                "measured": _round_float(measured),
                "threshold": _round_float(threshold),
                "comparison": ">=",
                "passed": measured >= threshold,
            }
        )

    auprc_multiplier_min = thresholds.get("auprc_multiplier_over_best_heuristic_min")
    if auprc_multiplier_min is not None:
        configured_gate_count += 1
        measured = float(measured_metrics["auprc_multiplier_over_best_heuristic"])
        threshold = float(auprc_multiplier_min)
        checks.append(
            {
                "name": "auprc_multiplier_over_best_heuristic",
                "measured": _round_float(measured),
                "threshold": _round_float(threshold),
                "comparison": ">=",
                "passed": measured >= threshold,
            }
        )

    generation_proxy_min = thresholds.get("generation_proxy_concentration_at_15_min")
    if generation_proxy_min is not None:
        configured_gate_count += 1
        measured = float(measured_metrics["generation_proxy_concentration_at_15"])
        threshold = float(generation_proxy_min)
        checks.append(
            {
                "name": "generation_proxy_concentration_at_15",
                "measured": _round_float(measured),
                "threshold": _round_float(threshold),
                "comparison": ">=",
                "passed": measured >= threshold,
            }
        )

    if configured_gate_count == 0:
        raise VerifierTrainingError(
            "Kill memo thresholds are not configured. Provide them in configs/kill_test.yaml or via CLI overrides."
        )

    decision = "GO" if all(bool(check["passed"]) for check in checks) else "NO_GO"
    memo_payload = {
        "version": KILL_MEMO_VERSION,
        "decision": decision,
        "best_model": {
            "checkpoint_filename": evaluation["best_model"]["checkpoint_filename"],
            "checkpoint_path": evaluation["best_model"]["checkpoint_path"],
            "model_type": evaluation["best_model"]["model_type"],
            "test_metrics": best_model_metrics,
        },
        "heuristic_baseline": {
            "signal_name": evaluation["heuristic_baseline"]["signal_name"],
            "direction_name": evaluation["heuristic_baseline"]["direction_name"],
            "test_metrics": heuristic_metrics,
        },
        "label_definition": label_definition,
        "failure_sparsity": failure_sparsity,
        "generation_proxy": {
            "budget_percent": LOCKED_KILL_GATE_PERCENT,
            "metric_name": GENERATION_PROXY_BUDGET_METRIC,
            "measured_concentration_at_15": measured_metrics["generation_proxy_concentration_at_15"],
        },
        "measured": measured_metrics,
        "thresholds": {
            "concentration_at_15_min": concentration_at_15_min,
            "gini_mean_min": gini_mean_min,
            "verifier_auprc_min": verifier_auprc_min,
            "auprc_multiplier_over_best_heuristic_min": auprc_multiplier_min,
            "generation_proxy_concentration_at_15_min": generation_proxy_min,
        },
        "checks": checks,
        "artifacts": {
            "eval_json": display_path(eval_json_path, repo_root),
            "budget_curves_csv": display_path(budget_curves_csv_path, repo_root),
            "error_map_root": display_path(error_map_root, repo_root),
        },
    }

    memo_md_lines = [
        "# Kill Test Memo",
        "",
        f"Decision: **{decision}**",
        "",
        f"Best model: `{evaluation['best_model']['checkpoint_filename']}`",
        f"Heuristic baseline: `{evaluation['heuristic_baseline']['signal_name']}` ({evaluation['heuristic_baseline']['direction_name']})",
        "",
        "## Measured Metrics",
        "",
        f"- Failure concentration at 15%: `{measured_metrics['failure_concentration_at_15']}`",
        f"- Failure Gini mean: `{measured_metrics['failure_gini_mean']}`",
        f"- Verifier test AUPRC: `{measured_metrics['verifier_test_auprc']}`",
        f"- Verifier test AUROC: `{measured_metrics['verifier_test_auroc']}`",
        f"- Best heuristic test AUPRC: `{measured_metrics['heuristic_test_auprc']}`",
        f"- Best heuristic test AUROC: `{measured_metrics['heuristic_test_auroc']}`",
        f"- AUPRC multiplier over best heuristic: `{measured_metrics['auprc_multiplier_over_best_heuristic']}`",
        (
            f"- Generation-proxy concentration at 15%: `{measured_metrics['generation_proxy_concentration_at_15']}` "
            f"(`{GENERATION_PROXY_BUDGET_METRIC}` at 15% patch budget)"
        ),
        "",
        "## Gate Checks",
        "",
        "| Check | Measured | Threshold | Result |",
        "| --- | ---: | ---: | --- |",
    ]
    for check in checks:
        memo_md_lines.append(
            f"| {check['name']} | {check['measured']} | {check['comparison']} {check['threshold']} | {_render_bool(bool(check['passed']))} |"
        )
    memo_md_lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Evaluation JSON: `{display_path(eval_json_path, repo_root)}`",
            f"- Budget curves CSV: `{display_path(budget_curves_csv_path, repo_root)}`",
            f"- Patch-error root: `{display_path(error_map_root, repo_root)}`",
        ]
    )

    memo_md_path.parent.mkdir(parents=True, exist_ok=True)
    memo_json_path.parent.mkdir(parents=True, exist_ok=True)
    with memo_md_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(memo_md_lines) + "\n")
    with memo_json_path.open("w", encoding="utf-8") as handle:
        json.dump(memo_payload, handle, indent=2, sort_keys=True)

    return {
        "memo_md_path": memo_md_path,
        "memo_json_path": memo_json_path,
        "payload": memo_payload,
    }
