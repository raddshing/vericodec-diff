from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.surrogate import DEFAULT_SURROGATE_FEATURE_NAMES, fit_linear_surrogate, score_records
from rd_lora.substrate.diffusers_sdxl import deep_update, load_yaml_mapping, save_json
from vericodec_diff.config import OmegaConf


DEFAULT_CONFIG_PATH = "configs/rdlora_vanilla.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the RD-LoRA surrogate and emit the standardized surrogate artifacts."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config path.")
    parser.add_argument("--probe_dir", required=True, help="Probe output directory with cell_utility.json.")
    parser.add_argument("--output_dir", required=True, help="Output directory for surrogate artifacts.")
    return parser.parse_args()


def _default_config() -> dict[str, Any]:
    return {
        "surrogate": {
            "feature_names": list(DEFAULT_SURROGATE_FEATURE_NAMES),
            "l2_regularization": 1.0e-6,
            "clip_min_utility": 0.0,
            "round_digits": 6,
            "val_modulus": 5,
        }
    }


def _load_config(path_value: str) -> dict[str, Any]:
    config_path = Path(path_value).expanduser()
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    raw = load_yaml_mapping(config_path) if config_path.is_file() else {}
    return deep_update(_default_config(), raw)


def _load_rows(probe_dir: Path) -> list[dict[str, Any]]:
    payload = load_yaml_mapping(probe_dir / "cell_utility.json")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{probe_dir / 'cell_utility.json'} must contain a non-empty rows list")
    return [dict(row) for row in rows]


def _split_rows(rows: Sequence[dict[str, Any]], *, val_modulus: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index % val_modulus == 0:
            val_rows.append(dict(row))
        else:
            train_rows.append(dict(row))
    if not train_rows:
        train_rows = [dict(row) for row in rows[:-1]]
        val_rows = [dict(rows[-1])]
    if not val_rows:
        val_rows = [dict(train_rows[-1])]
        train_rows = train_rows[:-1]
    return train_rows, val_rows


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and float(values[order[end]]) == float(values[order[position]]):
            end += 1
        average_rank = (position + end - 1) / 2.0 + 1.0
        for item_index in order[position:end]:
            ranks[item_index] = average_rank
        position = end
    return ranks


def _spearman_rho(targets: Sequence[float], predictions: Sequence[float]) -> float:
    if len(targets) != len(predictions) or not targets:
        raise ValueError("targets and predictions must be non-empty and aligned")
    target_ranks = np.asarray(_rankdata(targets), dtype=np.float64)
    prediction_ranks = np.asarray(_rankdata(predictions), dtype=np.float64)
    if np.std(target_ranks) == 0.0 or np.std(prediction_ranks) == 0.0:
        return 0.0
    return float(np.corrcoef(target_ranks, prediction_ranks)[0, 1])


def _top_k_precision(rows: Sequence[dict[str, Any]], *, k: int) -> float:
    if not rows:
        raise ValueError("rows must not be empty")
    top_k = min(int(k), len(rows))
    predicted = {
        (str(row["cell_id"]), int(row["candidate_rank"]))
        for row in sorted(rows, key=lambda item: (-float(item["predicted_utility"]), str(item["cell_id"])))[:top_k]
    }
    actual = {
        (str(row["cell_id"]), int(row["candidate_rank"]))
        for row in sorted(rows, key=lambda item: (-float(item["utility_score"]), str(item["cell_id"])))[:top_k]
    }
    return float(len(predicted & actual)) / float(top_k)


def main() -> int:
    args = parse_args()
    started_at = time.monotonic()
    config = _load_config(args.config)
    probe_dir = Path(args.probe_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config = deep_update(
        config,
        {
            "paths": {
                "repo_root": str(REPO_ROOT),
                "probe_dir": str(probe_dir),
                "output_dir": str(output_dir),
            }
        },
    )
    resolved_config_path = output_dir / "resolved_config.yaml"
    OmegaConf.save(OmegaConf.create(resolved_config), resolved_config_path)

    rows = _load_rows(probe_dir)
    train_rows, val_rows = _split_rows(rows, val_modulus=int(resolved_config["surrogate"]["val_modulus"]))
    fit_result = fit_linear_surrogate(
        train_rows,
        feature_names=resolved_config["surrogate"]["feature_names"],
        l2_regularization=float(resolved_config["surrogate"]["l2_regularization"]),
        clip_min_utility=resolved_config["surrogate"]["clip_min_utility"],
        round_digits=int(resolved_config["surrogate"]["round_digits"]),
    )
    model = fit_result["model"]
    scored_val_rows = score_records(model, val_rows)

    held_out_spearman_rho = round(
        _spearman_rho(
            [float(row["utility_score"]) for row in scored_val_rows],
            [float(row["predicted_utility"]) for row in scored_val_rows],
        ),
        6,
    )
    top_5_precision = round(_top_k_precision(scored_val_rows, k=5), 6)

    surrogate_model_path = output_dir / "surrogate_model.pkl"
    with surrogate_model_path.open("wb") as handle:
        pickle.dump(model, handle)

    surrogate_summary_path = output_dir / "surrogate_summary.json"
    save_json(
        surrogate_summary_path,
        {
            "schema_version": "1.0",
            "held_out_spearman_rho": held_out_spearman_rho,
            "top_5_precision": top_5_precision,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "source_probe_dir": str(probe_dir),
            "wall_time_seconds": round(time.monotonic() - started_at, 6),
        },
    )

    print(f"resolved_config={resolved_config_path}")
    print(f"surrogate_model={surrogate_model_path}")
    print(f"surrogate_summary={surrogate_summary_path}")
    print(f"held_out_spearman_rho={held_out_spearman_rho}")
    print(f"top_5_precision={top_5_precision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
