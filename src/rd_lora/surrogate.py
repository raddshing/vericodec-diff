from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rd_lora.features import UTILITY_RECORD_FIELDNAMES


DEFAULT_SURROGATE_TARGET = "utility_score"
DEFAULT_SURROGATE_FEATURE_NAMES = (
    "candidate_rank",
    "rank_fraction_of_max",
    "is_attention_cell",
    "layer_count",
    "step_count",
    "event_count",
    "baseline_mean_abs_drift",
    "mean_abs_drift",
    "rms_drift",
    "max_abs_drift",
    "cosine_distance",
    "baseline_attention_output_mean_abs_drift",
    "attention_output_mean_abs_drift",
    "attention_output_rms_drift",
    "attention_output_max_abs_drift",
    "attention_output_cosine_distance",
    "mean_abs_improvement",
    "attention_mean_abs_improvement",
    "has_attention_measurements",
)
SURROGATE_PREDICTION_FIELDNAMES = UTILITY_RECORD_FIELDNAMES + (
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


def sort_utility_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for record in records:
        payload = dict(record)
        missing = sorted(set(UTILITY_RECORD_FIELDNAMES) - set(payload))
        if missing:
            raise SurrogateValidationError(f"Utility record is missing fields: {missing}")
        normalized.append(payload)
    return sorted(
        normalized,
        key=lambda item: (
            str(item["cell_id"]),
            int(item["candidate_rank"]),
            str(item["layer_group_id"]),
            str(item["timestep_band_id"]),
        ),
    )


def load_utility_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SurrogateValidationError(f"{path} must parse to a mapping")
    records = payload.get("records")
    if not isinstance(records, list):
        raise SurrogateValidationError(f"{path} must define a records list")
    return sort_utility_records(records)


def _feature_value(record: Mapping[str, Any], feature_name: str) -> float:
    if feature_name == "mean_abs_improvement":
        return _as_float(record["baseline_mean_abs_drift"]) - _as_float(record["mean_abs_drift"])
    if feature_name == "attention_mean_abs_improvement":
        return _as_float(record["baseline_attention_output_mean_abs_drift"]) - _as_float(
            record["attention_output_mean_abs_drift"]
        )
    if feature_name == "has_attention_measurements":
        return 1.0 if record.get("attention_output_mean_abs_drift") is not None else 0.0
    if feature_name not in record:
        raise SurrogateValidationError(f"Unknown surrogate feature {feature_name!r}")
    return _as_float(record[feature_name])


def build_feature_vector(record: Mapping[str, Any], feature_names: Sequence[str]) -> tuple[float, ...]:
    return tuple(_feature_value(record, feature_name) for feature_name in feature_names)


def build_feature_matrix(records: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([build_feature_vector(record, feature_names) for record in records], dtype=np.float64)


def build_target_vector(records: Sequence[Mapping[str, Any]], target_name: str) -> np.ndarray:
    if target_name != DEFAULT_SURROGATE_TARGET:
        raise SurrogateValidationError(f"Unsupported surrogate target {target_name!r}")
    return np.asarray([_as_float(record[target_name]) for record in records], dtype=np.float64)


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
    "SURROGATE_METRIC_FIELDNAMES",
    "SURROGATE_PREDICTION_FIELDNAMES",
    "SurrogateValidationError",
    "build_feature_matrix",
    "build_feature_vector",
    "compute_prediction_metrics",
    "fit_linear_surrogate",
    "load_utility_records",
    "score_records",
    "sort_utility_records",
    "write_metric_csv",
    "write_prediction_csv",
]
