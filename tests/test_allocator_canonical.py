from __future__ import annotations

import csv
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.allocator import (
    _solve_layer_only,
    _solve_proposed,
    _solve_timestep_only,
    _solve_uniform,
    build_allocation_manifest,
    canonicalize_scored_rows,
)
from rd_lora.cells import build_cell_schema
from rd_lora.runtime.allocation_manifest import load_allocation_manifest, validate_allocation_manifest
from rd_lora.surrogate import DEFAULT_SURROGATE_FEATURE_NAMES, LinearUtilitySurrogate


SOLVE_SCRIPT = REPO_ROOT / "scripts" / "solve_rdlora_allocation.py"
CANONICAL_KEYS = {
    "cell_id",
    "layer_group",
    "timestep_band",
    "candidate_rank",
    "predicted_utility",
}
RANK_BUDGET_TOTAL = 48


@pytest.fixture
def schema():
    return build_cell_schema(candidate_ranks=(0, 2, 4))


@pytest.fixture
def surrogate_model():
    coefficients = []
    for feature_name in DEFAULT_SURROGATE_FEATURE_NAMES:
        if feature_name == "candidate_rank":
            coefficients.append(0.8)
        elif feature_name == "layer_group_id":
            coefficients.append(-0.8)
        elif feature_name == "timestep_band_id":
            coefficients.append(1.2)
        else:
            coefficients.append(0.0)
    return LinearUtilitySurrogate(
        feature_names=tuple(DEFAULT_SURROGATE_FEATURE_NAMES),
        coefficients=tuple(coefficients),
        bias=-1.0,
        target_name="utility",
        l2_regularization=0.0,
        clip_min_utility=None,
        round_digits=6,
    )


@pytest.fixture
def scored_rows(schema, surrogate_model):
    rows: list[dict[str, object]] = []
    max_rank = max(int(rank) for rank in schema.candidate_ranks)
    for cell in schema.cells:
        for candidate_rank in schema.candidate_ranks:
            candidate_rank = int(candidate_rank)
            pre_loss = round(1.0 + (0.01 * cell.cell_index) + (0.001 * candidate_rank), 6)
            utility = round((0.12 * candidate_rank) + (0.015 * cell.cell_index), 6)
            post_loss = round(pre_loss - utility, 6)
            row: dict[str, object] = {
                "candidate_rank": candidate_rank,
                "cell_id": cell.cell_id,
                "layer_group": cell.layer_group.group_id,
                "layer_group_id": cell.layer_group.group_index,
                "optimizer_steps": 1,
                "post_loss": post_loss,
                "pre_loss": pre_loss,
                "rank_fraction_of_max": 0.0 if max_rank <= 0 else candidate_rank / float(max_rank),
                "task": "subject_personalization",
                "task_id": 0,
                "timestep_band": cell.timestep_band.band_id,
                "timestep_band_id": cell.timestep_band.band_index,
                "train_batch_count": 1,
                "utility": utility,
                "utility_score": utility,
                "val_batch_count": 1,
            }
            predicted_utility = surrogate_model.predict_record(row)
            row["predicted_utility"] = predicted_utility
            row["residual_utility"] = round(float(predicted_utility) - float(utility), 6)
            rows.append(row)
    return rows


def _write_probe_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_canonicalization_succeeds(schema, scored_rows) -> None:
    rows = [dict(row) for row in scored_rows]
    rows[0]["cell_id"] = ""
    rows[0]["layer_group"] = rows[0]["layer_group_id"]
    rows[0]["timestep_band"] = rows[0]["timestep_band_id"]

    canonical_rows = canonicalize_scored_rows(rows, schema=schema)

    assert len(canonical_rows) == len(scored_rows)
    assert all(set(row) == CANONICAL_KEYS for row in canonical_rows)
    assert canonical_rows[0]["cell_id"] == "layer_group_00__timestep_band_00"
    assert canonical_rows[0]["layer_group"] == "layer_group_00"
    assert canonical_rows[0]["timestep_band"] == "timestep_band_00"


def test_solve_proposed_never_touches_missing_count_fields(schema, scored_rows) -> None:
    canonical_rows = canonicalize_scored_rows(scored_rows, schema=schema)

    proposed_ranks = _solve_proposed(
        canonical_rows,
        schema=schema,
        rank_budget_total=RANK_BUDGET_TOTAL,
    )

    assert len(proposed_ranks) == len(schema.cells)
    assert any(rank > 0 for rank in proposed_ranks.values())


def test_all_four_manifests_validate_and_proposed_differs(schema, scored_rows) -> None:
    canonical_rows = canonicalize_scored_rows(scored_rows, schema=schema)

    uniform_ranks = _solve_uniform(
        canonical_rows,
        schema=schema,
        rank_budget_total=RANK_BUDGET_TOTAL,
    )
    layer_only_ranks = _solve_layer_only(
        canonical_rows,
        schema=schema,
        rank_budget_total=RANK_BUDGET_TOTAL,
    )
    timestep_only_ranks = _solve_timestep_only(
        canonical_rows,
        schema=schema,
        rank_budget_total=RANK_BUDGET_TOTAL,
    )
    proposed_ranks = _solve_proposed(
        canonical_rows,
        schema=schema,
        rank_budget_total=RANK_BUDGET_TOTAL,
    )

    manifests = {
        "uniform": build_allocation_manifest(
            "uniform",
            schema=schema,
            rank_budget_total=RANK_BUDGET_TOTAL,
            cell_ranks=uniform_ranks,
        ),
        "layer_only": build_allocation_manifest(
            "layer_only",
            schema=schema,
            rank_budget_total=RANK_BUDGET_TOTAL,
            cell_ranks=layer_only_ranks,
        ),
        "timestep_only": build_allocation_manifest(
            "timestep_only",
            schema=schema,
            rank_budget_total=RANK_BUDGET_TOTAL,
            cell_ranks=timestep_only_ranks,
        ),
        "proposed": build_allocation_manifest(
            "proposed",
            schema=schema,
            rank_budget_total=RANK_BUDGET_TOTAL,
            cell_ranks=proposed_ranks,
        ),
    }

    for manifest in manifests.values():
        validate_allocation_manifest(manifest)

    assert proposed_ranks != uniform_ranks


def test_solve_script_writes_required_outputs(tmp_path: Path, scored_rows, surrogate_model) -> None:
    probe_dir = tmp_path / "probe"
    surrogate_dir = tmp_path / "surrogate"
    output_dir = tmp_path / "allocation"
    probe_dir.mkdir()
    surrogate_dir.mkdir()

    _write_probe_csv(probe_dir / "cell_utility.csv", scored_rows)
    with (surrogate_dir / "surrogate_model.pkl").open("wb") as handle:
        pickle.dump(surrogate_model, handle)

    config_path = tmp_path / "allocator.yaml"
    config_path.write_text(
        "\n".join(
            [
                "allocation:",
                f"  rank_budget_total: {RANK_BUDGET_TOTAL}",
                "  utility_scale: 1000000",
                "  max_exact_state_count: 500000",
                "training:",
                "  target_modules:",
                "    - to_k",
                "    - to_q",
                "    - to_v",
                "    - to_out.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    subprocess.run(
        [
            sys.executable,
            str(SOLVE_SCRIPT),
            "--config",
            str(config_path),
            "--probe_dir",
            str(probe_dir),
            "--surrogate_dir",
            str(surrogate_dir),
            "--output_dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((output_dir / "allocation_summary.json").read_text(encoding="utf-8"))
    assert {"rank_budget_total", "manifest_files", "backends"} <= set(summary)
    assert summary["rank_budget_total"] == RANK_BUDGET_TOTAL
    assert summary["backends"] == ["uniform", "layer_only", "timestep_only", "proposed"]
    assert (output_dir / "resolved_config.yaml").is_file()

    manifests = {
        backend: load_allocation_manifest(output_dir / f"{backend}.json")
        for backend in summary["backends"]
    }
    assert set(manifests) == {"uniform", "layer_only", "timestep_only", "proposed"}
    assert [cell["rank"] for cell in manifests["proposed"]["cells"]] != [
        cell["rank"] for cell in manifests["uniform"]["cells"]
    ]
