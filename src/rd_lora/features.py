from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


UTILITY_RECORD_FIELDNAMES = (
    "cell_id",
    "layer_group_id",
    "timestep_band_id",
    "candidate_rank",
    "rank_fraction_of_max",
    "is_attention_cell",
    "layer_count",
    "step_count",
    "event_count",
    "baseline_mean_abs_drift",
    "baseline_attention_output_mean_abs_drift",
    "mean_abs_drift",
    "rms_drift",
    "max_abs_drift",
    "cosine_distance",
    "attention_output_mean_abs_drift",
    "attention_output_rms_drift",
    "attention_output_max_abs_drift",
    "attention_output_cosine_distance",
    "utility_score",
)


def _round_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _vector_drift(reference: Sequence[float], candidate: Sequence[float]) -> dict[str, float]:
    if len(reference) != len(candidate):
        raise ValueError("reference and candidate vectors must have the same length")
    if not reference:
        raise ValueError("reference and candidate vectors must be non-empty")

    deltas = [float(candidate[index]) - float(reference[index]) for index in range(len(reference))]
    abs_deltas = [abs(value) for value in deltas]
    mean_abs = sum(abs_deltas) / len(abs_deltas)
    rms = math.sqrt(sum(value * value for value in deltas) / len(deltas))
    max_abs = max(abs_deltas)

    ref_norm = math.sqrt(sum(float(value) * float(value) for value in reference))
    cand_norm = math.sqrt(sum(float(value) * float(value) for value in candidate))
    if ref_norm == 0.0 or cand_norm == 0.0:
        cosine_distance = 0.0
    else:
        dot = sum(float(reference[index]) * float(candidate[index]) for index in range(len(reference)))
        cosine_similarity = max(-1.0, min(1.0, dot / (ref_norm * cand_norm)))
        cosine_distance = 1.0 - cosine_similarity

    return {
        "mean_abs_drift": mean_abs,
        "rms_drift": rms,
        "max_abs_drift": max_abs,
        "cosine_distance": cosine_distance,
    }


def summarize_ranked_events(
    events: Sequence[Mapping[str, Any]],
    *,
    candidate_rank: int,
    include_attention_output_drift: bool,
) -> dict[str, Any]:
    if not events:
        raise ValueError("events must not be empty")

    mean_abs_values: list[float] = []
    rms_values: list[float] = []
    max_abs_values: list[float] = []
    cosine_values: list[float] = []
    attention_mean_abs_values: list[float] = []
    attention_rms_values: list[float] = []
    attention_max_abs_values: list[float] = []
    attention_cosine_values: list[float] = []

    for event in events:
        reference_output = event["reference_output"]
        candidate_output = event["candidate_outputs"][candidate_rank]
        drift = _vector_drift(reference_output, candidate_output)
        mean_abs_values.append(drift["mean_abs_drift"])
        rms_values.append(drift["rms_drift"])
        max_abs_values.append(drift["max_abs_drift"])
        cosine_values.append(drift["cosine_distance"])
        if include_attention_output_drift and str(event.get("layer_kind")) == "attention":
            attention_mean_abs_values.append(drift["mean_abs_drift"])
            attention_rms_values.append(drift["rms_drift"])
            attention_max_abs_values.append(drift["max_abs_drift"])
            attention_cosine_values.append(drift["cosine_distance"])

    attention_enabled = bool(include_attention_output_drift and attention_mean_abs_values)
    return {
        "event_count": len(events),
        "mean_abs_drift": sum(mean_abs_values) / len(mean_abs_values),
        "rms_drift": sum(rms_values) / len(rms_values),
        "max_abs_drift": max(max_abs_values),
        "cosine_distance": sum(cosine_values) / len(cosine_values),
        "attention_output_mean_abs_drift": (
            sum(attention_mean_abs_values) / len(attention_mean_abs_values) if attention_enabled else None
        ),
        "attention_output_rms_drift": sum(attention_rms_values) / len(attention_rms_values) if attention_enabled else None,
        "attention_output_max_abs_drift": max(attention_max_abs_values) if attention_enabled else None,
        "attention_output_cosine_distance": (
            sum(attention_cosine_values) / len(attention_cosine_values) if attention_enabled else None
        ),
    }


def build_cell_utility_records(
    *,
    cell: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    candidate_ranks: Sequence[int],
    include_attention_output_drift: bool,
) -> list[dict[str, Any]]:
    ranks = tuple(int(rank) for rank in candidate_ranks)
    if not ranks:
        raise ValueError("candidate_ranks must not be empty")
    if ranks[0] != 0:
        raise ValueError("candidate_ranks must start with rank 0")

    baseline = summarize_ranked_events(
        events,
        candidate_rank=0,
        include_attention_output_drift=include_attention_output_drift,
    )
    max_rank = max(ranks)
    records: list[dict[str, Any]] = []
    for rank in ranks:
        summary = summarize_ranked_events(
            events,
            candidate_rank=rank,
            include_attention_output_drift=include_attention_output_drift,
        )
        baseline_attention = baseline["attention_output_mean_abs_drift"] or 0.0
        current_attention = summary["attention_output_mean_abs_drift"] or 0.0
        utility_score = max(0.0, baseline["mean_abs_drift"] - summary["mean_abs_drift"])
        utility_score += 0.5 * max(0.0, baseline_attention - current_attention)
        rank_fraction = float(rank) / float(max_rank) if max_rank > 0 else 0.0

        records.append(
            {
                "cell_id": str(cell["cell_id"]),
                "layer_group_id": str(cell["layer_group_id"]),
                "timestep_band_id": str(cell["timestep_band_id"]),
                "candidate_rank": rank,
                "rank_fraction_of_max": _round_float(rank_fraction),
                "is_attention_cell": bool(cell["is_attention_cell"]),
                "layer_count": int(cell["layer_count"]),
                "step_count": int(cell["step_count"]),
                "event_count": int(summary["event_count"]),
                "baseline_mean_abs_drift": _round_float(baseline["mean_abs_drift"]),
                "baseline_attention_output_mean_abs_drift": _round_float(
                    baseline["attention_output_mean_abs_drift"]
                ),
                "mean_abs_drift": _round_float(summary["mean_abs_drift"]),
                "rms_drift": _round_float(summary["rms_drift"]),
                "max_abs_drift": _round_float(summary["max_abs_drift"]),
                "cosine_distance": _round_float(summary["cosine_distance"]),
                "attention_output_mean_abs_drift": _round_float(summary["attention_output_mean_abs_drift"]),
                "attention_output_rms_drift": _round_float(summary["attention_output_rms_drift"]),
                "attention_output_max_abs_drift": _round_float(summary["attention_output_max_abs_drift"]),
                "attention_output_cosine_distance": _round_float(summary["attention_output_cosine_distance"]),
                "utility_score": _round_float(utility_score),
            }
        )
    return records


__all__ = [
    "UTILITY_RECORD_FIELDNAMES",
    "build_cell_utility_records",
    "summarize_ranked_events",
]
