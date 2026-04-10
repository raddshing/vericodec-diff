from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


REQUIRED_PROBE_COLUMNS = (
    "task",
    "cell_id",
    "layer_group",
    "timestep_band",
    "candidate_rank",
    "pre_loss",
    "post_loss",
    "utility",
    "optimizer_steps",
    "train_batch_count",
    "val_batch_count",
)
SURROGATE_FEATURES_V1 = [
    "candidate_rank",
    "rank_fraction_of_max",
    "optimizer_steps",
    "train_batch_count",
    "val_batch_count",
    "pre_loss",
    "layer_group_id",
    "timestep_band_id",
    "task_id",
]
DEFAULT_SURROGATE_TARGET = "utility"
DEFAULT_SURROGATE_FEATURE_NAMES = tuple(SURROGATE_FEATURES_V1)
SURROGATE_PREDICTION_FIELDNAMES = REQUIRED_PROBE_COLUMNS + (
    "utility_score",
    "rank_fraction_of_max",
    "layer_group_id",
    "timestep_band_id",
    "task_id",
    "predicted_utility",
    "residual_utility",
)
SURROGATE_METRIC_FIELDNAMES = (
    "record_count",
    "mae",
    "rmse",
    "max_abs_error",
    "mean_target",
    "mean_prediction",
)


class SurrogateValidationError(ValueError):
    """Raised when surrogate inputs or artifacts are malformed."""


def _round_float(value: float, *, digits: int = 6) -> float:
    return round(float(value), int(digits))


def _as_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def _resolve_probe_artifact_paths(probe_path: Path) -> tuple[Path, Path]:
    resolved = probe_path.expanduser()
    if resolved.is_dir() or not resolved.suffix:
        return resolved / "cell_utility.csv", resolved / "cell_utility.json"
    if resolved.suffix == ".csv":
        return resolved, resolved.with_name("cell_utility.json")
    if resolved.suffix == ".json":
        return resolved.with_name("cell_utility.csv"), resolved
    return resolved / "cell_utility.csv", resolved / "cell_utility.json"


def _read_probe_dataframe(probe_path: Path) -> pd.DataFrame:
    csv_path, json_path = _resolve_probe_artifact_paths(probe_path)
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise SurrogateValidationError(f"{json_path} must parse to a mapping")
        rows = payload.get("rows")
        if rows is None:
            rows = payload.get("records")
        if not isinstance(rows, list):
            raise SurrogateValidationError(f"{json_path} must define a rows or records list")
        return pd.DataFrame(rows)
    raise FileNotFoundError(f"Missing {csv_path.name} and {json_path.name} under {csv_path.parent}")


def _ensure_dataframe(records: Sequence[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    return pd.DataFrame([dict(record) for record in records])


def _encode_categorical(values: pd.Series) -> pd.Series:
    string_values = values.astype(str)
    categories = sorted(string_values.unique().tolist())
    return pd.Series(
        pd.Categorical(string_values, categories=categories, ordered=True).codes,
        index=values.index,
        dtype=np.int64,
    )


def normalize_probe_dataframe(
    frame: pd.DataFrame,
    *,
    force_recompute: bool = False,
) -> pd.DataFrame:
    normalized = frame.copy()
    if normalized.empty:
        raise SurrogateValidationError("Probe artifact contains no rows")

    if "utility" not in normalized.columns and "utility_score" in normalized.columns:
        normalized["utility"] = normalized["utility_score"]

    missing = sorted(set(REQUIRED_PROBE_COLUMNS) - set(normalized.columns))
    if missing:
        raise SurrogateValidationError(f"Missing required probe columns: {missing}")

    for column in ("task", "cell_id", "layer_group", "timestep_band"):
        normalized[column] = normalized[column].astype(str)

    integer_columns = (
        "candidate_rank",
        "optimizer_steps",
        "train_batch_count",
        "val_batch_count",
    )
    float_columns = ("pre_loss", "post_loss", "utility")

    for column in integer_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype(np.int64)
    for column in float_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype(np.float64)

    max_candidate_rank = max(int(normalized["candidate_rank"].max()), 1)
    if force_recompute or "rank_fraction_of_max" not in normalized.columns:
        normalized["rank_fraction_of_max"] = normalized["candidate_rank"].astype(np.float64) / float(max_candidate_rank)
    else:
        normalized["rank_fraction_of_max"] = pd.to_numeric(
            normalized["rank_fraction_of_max"],
            errors="raise",
        ).astype(np.float64)

    derived_categorical_columns = (
        ("layer_group", "layer_group_id"),
        ("timestep_band", "timestep_band_id"),
        ("task", "task_id"),
    )
    for source_column, target_column in derived_categorical_columns:
        should_recompute = (
            force_recompute
            or target_column not in normalized.columns
            or not pd.api.types.is_numeric_dtype(normalized[target_column])
        )
        if should_recompute:
            normalized[target_column] = _encode_categorical(normalized[source_column])
        else:
            normalized[target_column] = pd.to_numeric(normalized[target_column], errors="raise").astype(np.int64)

    normalized["utility_score"] = normalized["utility"].astype(np.float64)

    return normalized.sort_values(
        by=["task", "cell_id", "candidate_rank", "layer_group_id", "timestep_band_id"],
        kind="stable",
    ).reset_index(drop=True)


def load_probe_dataframe(probe_path: Path) -> pd.DataFrame:
    return normalize_probe_dataframe(_read_probe_dataframe(probe_path), force_recompute=True)


def sort_utility_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = normalize_probe_dataframe(_ensure_dataframe(records))
    return normalized.to_dict(orient="records")


def load_utility_records(path: Path) -> list[dict[str, Any]]:
    return load_probe_dataframe(path).to_dict(orient="records")


def build_feature_vector(record: Mapping[str, Any], feature_names: Sequence[str]) -> tuple[float, ...]:
    values: list[float] = []
    for feature_name in feature_names:
        if feature_name not in record:
            raise SurrogateValidationError(f"Unknown surrogate feature {feature_name!r}")
        values.append(_as_float(record[feature_name]))
    return tuple(values)


def build_feature_matrix(
    records: Sequence[Mapping[str, Any]] | pd.DataFrame,
    feature_names: Sequence[str],
) -> np.ndarray:
    normalized = normalize_probe_dataframe(_ensure_dataframe(records))
    missing = [feature_name for feature_name in feature_names if feature_name not in normalized.columns]
    if missing:
        raise SurrogateValidationError(f"Unknown surrogate features: {missing}")
    return normalized.loc[:, list(feature_names)].to_numpy(dtype=np.float64)


def build_target_vector(
    records: Sequence[Mapping[str, Any]] | pd.DataFrame,
    target_name: str,
) -> np.ndarray:
    if target_name != DEFAULT_SURROGATE_TARGET:
        raise SurrogateValidationError(f"Unsupported surrogate target {target_name!r}")
    normalized = normalize_probe_dataframe(_ensure_dataframe(records))
    return normalized[target_name].to_numpy(dtype=np.float64)


@dataclass(frozen=True)
class LinearUtilitySurrogate:
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    bias: float
    target_name: str
    l2_regularization: float
    clip_min_utility: float | None
    round_digits: int

    def predict_record(self, record: Mapping[str, Any]) -> float:
        if int(record["candidate_rank"]) == 0:
            return 0.0
        vector = build_feature_vector(record, self.feature_names)
        value = float(self.bias)
        for coefficient, feature_value in zip(self.coefficients, vector):
            value += float(coefficient) * float(feature_value)
        if self.clip_min_utility is not None:
            value = max(float(self.clip_min_utility), value)
        return _round_float(value, digits=self.round_digits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_type": "linear_ridge",
            "feature_names": list(self.feature_names),
            "coefficients": [_round_float(value, digits=12) for value in self.coefficients],
            "bias": _round_float(self.bias, digits=12),
            "target_name": self.target_name,
            "l2_regularization": _round_float(self.l2_regularization, digits=12),
            "clip_min_utility": self.clip_min_utility,
            "round_digits": self.round_digits,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LinearUtilitySurrogate":
        feature_names = tuple(str(value) for value in payload.get("feature_names", []))
        coefficients = tuple(float(value) for value in payload.get("coefficients", []))
        if len(feature_names) != len(coefficients):
            raise SurrogateValidationError("Surrogate model feature_names and coefficients must match in length")
        return cls(
            feature_names=feature_names,
            coefficients=coefficients,
            bias=float(payload.get("bias", 0.0)),
            target_name=str(payload.get("target_name", DEFAULT_SURROGATE_TARGET)),
            l2_regularization=float(payload.get("l2_regularization", 0.0)),
            clip_min_utility=(None if payload.get("clip_min_utility") is None else float(payload["clip_min_utility"])),
            round_digits=int(payload.get("round_digits", 6)),
        )


def compute_prediction_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    target_name: str = DEFAULT_SURROGATE_TARGET,
    prediction_name: str = "predicted_utility",
) -> dict[str, Any]:
    if not records:
        raise SurrogateValidationError("At least one surrogate prediction record is required")
    targets = np.asarray([_as_float(record[target_name]) for record in records], dtype=np.float64)
    predictions = np.asarray([_as_float(record[prediction_name]) for record in records], dtype=np.float64)
    residuals = predictions - targets
    abs_residuals = np.abs(residuals)
    return {
        "record_count": len(records),
        "mae": _round_float(float(np.mean(abs_residuals))),
        "rmse": _round_float(math.sqrt(float(np.mean(np.square(residuals))))),
        "max_abs_error": _round_float(float(np.max(abs_residuals))),
        "mean_target": _round_float(float(np.mean(targets))),
        "mean_prediction": _round_float(float(np.mean(predictions))),
    }


def score_records(
    model: LinearUtilitySurrogate,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    scored_records: list[dict[str, Any]] = []
    for record in sort_utility_records(records):
        payload = dict(record)
        predicted = model.predict_record(payload)
        payload["predicted_utility"] = predicted
        payload["residual_utility"] = _round_float(predicted - _as_float(payload[model.target_name]))
        scored_records.append(payload)
    return scored_records


def fit_linear_surrogate(
    records: Sequence[Mapping[str, Any]],
    *,
    feature_names: Sequence[str] = DEFAULT_SURROGATE_FEATURE_NAMES,
    target_name: str = DEFAULT_SURROGATE_TARGET,
    l2_regularization: float = 1.0e-6,
    clip_min_utility: float | None = 0.0,
    round_digits: int = 6,
) -> dict[str, Any]:
    sorted_records = sort_utility_records(records)
    if not sorted_records:
        raise SurrogateValidationError("At least one utility record is required")

    normalized_feature_names = tuple(str(value) for value in feature_names)
    if not normalized_feature_names:
        raise SurrogateValidationError("surrogate.feature_names must not be empty")

    l2_value = float(l2_regularization)
    if l2_value < 0.0:
        raise SurrogateValidationError("surrogate.l2_regularization must be non-negative")

    feature_matrix = build_feature_matrix(sorted_records, normalized_feature_names)
    target_vector = build_target_vector(sorted_records, target_name)
    design_matrix = np.concatenate(
        [np.ones((feature_matrix.shape[0], 1), dtype=np.float64), feature_matrix],
        axis=1,
    )
    gram = design_matrix.T @ design_matrix
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * l2_value
    regularizer[0, 0] = 0.0
    rhs = design_matrix.T @ target_vector
    try:
        weights = np.linalg.solve(gram + regularizer, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gram + regularizer) @ rhs

    model = LinearUtilitySurrogate(
        feature_names=normalized_feature_names,
        coefficients=tuple(float(value) for value in weights[1:]),
        bias=float(weights[0]),
        target_name=target_name,
        l2_regularization=l2_value,
        clip_min_utility=(None if clip_min_utility is None else float(clip_min_utility)),
        round_digits=int(round_digits),
    )
    scored_records = score_records(model, sorted_records)
    return {
        "model": model,
        "records": scored_records,
        "metrics": compute_prediction_metrics(scored_records, target_name=target_name),
    }


def write_prediction_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SURROGATE_PREDICTION_FIELDNAMES))
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in SURROGATE_PREDICTION_FIELDNAMES})


def write_metric_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SURROGATE_METRIC_FIELDNAMES))
        writer.writeheader()
        writer.writerow({field: metrics.get(field) for field in SURROGATE_METRIC_FIELDNAMES})


__all__ = [
    "DEFAULT_SURROGATE_FEATURE_NAMES",
    "DEFAULT_SURROGATE_TARGET",
    "LinearUtilitySurrogate",
    "REQUIRED_PROBE_COLUMNS",
    "SURROGATE_FEATURES_V1",
    "SURROGATE_METRIC_FIELDNAMES",
    "SURROGATE_PREDICTION_FIELDNAMES",
    "SurrogateValidationError",
    "build_feature_matrix",
    "build_feature_vector",
    "build_target_vector",
    "compute_prediction_metrics",
    "fit_linear_surrogate",
    "load_probe_dataframe",
    "load_utility_records",
    "normalize_probe_dataframe",
    "score_records",
    "sort_utility_records",
    "write_metric_csv",
    "write_prediction_csv",
]
