from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.cells import (
    DEFAULT_CANDIDATE_RANKS,
    DEFAULT_TARGET_MODULES,
    build_cell_schema,
    discover_valid_attention_blocks,
    parse_candidate_ranks_argument,
    validate_cell_targets,
    validate_layer_groups,
    validate_layer_group_inventory,
)
from rd_lora.features import (
    REQUIRED_PROBE_ROW_FIELDNAMES,
    REQUIRED_PROBE_SUMMARY_KEYS,
    REQUIRED_PROVENANCE_KEYS,
    UTILITY_RECORD_FIELDNAMES,
)


RUN_SCRIPT = REPO_ROOT / "scripts" / "run_rdlora_probe.py"
EXPECTED_LAYER_GROUPS = {
    "layer_group_00": ("down_blocks.1.attentions.0", "down_blocks.1.attentions.1"),
    "layer_group_01": ("down_blocks.2.attentions.0", "down_blocks.2.attentions.1"),
    "layer_group_02": ("mid_block.attentions.0",),
    "layer_group_03": ("up_blocks.0.attentions.0", "up_blocks.0.attentions.1"),
    "layer_group_04": ("up_blocks.0.attentions.2", "up_blocks.1.attentions.0"),
    "layer_group_05": ("up_blocks.1.attentions.1", "up_blocks.1.attentions.2"),
}
VALID_SDXL_ATTENTION_BLOCKS = tuple(
    layer_id for layer_ids in EXPECTED_LAYER_GROUPS.values() for layer_id in layer_ids
)
ALL_SDXL_ATTENTION_BLOCKS = (
    "down_blocks.0.attentions.0",
    "down_blocks.0.attentions.1",
    "down_blocks.1.attentions.0",
    "down_blocks.1.attentions.1",
    "down_blocks.2.attentions.0",
    "down_blocks.2.attentions.1",
    "mid_block.attentions.0",
    "mid_block.attentions.1",
    "up_blocks.0.attentions.0",
    "up_blocks.0.attentions.1",
    "up_blocks.0.attentions.2",
    "up_blocks.1.attentions.0",
    "up_blocks.1.attentions.1",
    "up_blocks.1.attentions.2",
    "up_blocks.2.attentions.0",
    "up_blocks.2.attentions.1",
)


class FakeUnet:
    def __init__(self, module_names: list[str]) -> None:
        self._module_names = tuple(module_names)

    def named_modules(self):  # type: ignore[no-untyped-def]
        yield "", self
        for module_name in self._module_names:
            yield module_name, object()


def _build_fake_unet(*, excluded_blocks: tuple[str, ...] = ()) -> FakeUnet:
    module_names: list[str] = []
    excluded = set(excluded_blocks)
    for block_name in VALID_SDXL_ATTENTION_BLOCKS:
        if block_name in excluded:
            continue
        module_names.extend(
            f"{block_name}.transformer_blocks.0.attn1.{target_module}" for target_module in DEFAULT_TARGET_MODULES
        )
    return FakeUnet(module_names)


def test_cell_schema_is_deterministic_and_has_24_cells() -> None:
    schema_a = build_cell_schema()
    schema_b = build_cell_schema()
    actual_groups = {group.group_id: group.layer_ids for group in schema_a.layer_groups}
    grouped_layer_ids = {layer_id for layer_ids in actual_groups.values() for layer_id in layer_ids}

    assert schema_a.to_dict() == schema_b.to_dict()
    assert len(schema_a.layer_groups) == 6
    assert len(schema_a.timestep_bands) == 4
    assert len(schema_a.cells) == 24
    assert schema_a.candidate_ranks == DEFAULT_CANDIDATE_RANKS
    assert actual_groups == EXPECTED_LAYER_GROUPS
    assert all(group.layer_ids for group in schema_a.layer_groups)
    assert "down_blocks.0.attentions.0" not in grouped_layer_ids
    assert "down_blocks.0.attentions.1" not in grouped_layer_ids
    assert "mid_block.attentions.1" not in grouped_layer_ids
    assert "up_blocks.2.attentions.0" not in grouped_layer_ids


def test_candidate_rank_parsing_is_deterministic() -> None:
    assert parse_candidate_ranks_argument("16, 4, 0, 8, 2, 8") == DEFAULT_CANDIDATE_RANKS
    assert parse_candidate_ranks_argument((0, 16, 2, 4, 8, 2)) == DEFAULT_CANDIDATE_RANKS


def test_discover_valid_attention_blocks_returns_11_blocks_for_fake_sdxl_unet() -> None:
    discovered_blocks = discover_valid_attention_blocks(_build_fake_unet())

    assert discovered_blocks == VALID_SDXL_ATTENTION_BLOCKS
    assert len(discovered_blocks) == 11
    assert set(discovered_blocks).issubset(set(ALL_SDXL_ATTENTION_BLOCKS))


def test_inventory_and_cell_target_validation_match_grouped_blocks() -> None:
    schema = build_cell_schema()
    unet = _build_fake_unet()

    discovered_blocks = discover_valid_attention_blocks(unet)
    validate_layer_groups(schema.layer_groups, discovered_blocks)
    inventory = validate_layer_group_inventory(unet)
    matches_by_cell = validate_cell_targets(schema, unet, DEFAULT_TARGET_MODULES)

    assert discovered_blocks == VALID_SDXL_ATTENTION_BLOCKS
    assert set(inventory) == set(VALID_SDXL_ATTENTION_BLOCKS)
    assert len(matches_by_cell) == 24


def test_cell_target_validation_rejects_empty_group_matches() -> None:
    schema = build_cell_schema()
    unet = _build_fake_unet(excluded_blocks=EXPECTED_LAYER_GROUPS["layer_group_05"])

    try:
        validate_cell_targets(schema, unet, DEFAULT_TARGET_MODULES)
    except ValueError as exc:
        assert "layer_group_05__timestep_band_00" in str(exc)
    else:
        raise AssertionError("validate_cell_targets should reject cells that match zero UNet modules")


def test_probe_output_schema_matches_required_columns_and_keys(tmp_path: Path) -> None:
    output_dir = tmp_path / "probe"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_SCRIPT),
            "--config",
            "configs/rdlora_probe.yaml",
            "--task",
            "subject_personalization",
            "--run_mode",
            "cpu_mock",
            "--output_dir",
            str(output_dir),
            "--max_cells",
            "2",
            "--candidate_ranks",
            "0,2",
            "--probe_inner_steps",
            "1",
            "--max_train_batches",
            "1",
            "--max_val_batches",
            "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "resolved_config.yaml").is_file()

    payload = json.loads((output_dir / "cell_utility.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "probe_summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((output_dir / "run_provenance.json").read_text(encoding="utf-8"))

    assert payload["row_count"] == 4
    assert summary["probe_mode"] == "cpu_mock"
    assert summary["row_count"] == 4
    assert summary["cell_count"] == 2
    assert set(REQUIRED_PROBE_SUMMARY_KEYS).issubset(summary)
    assert set(REQUIRED_PROVENANCE_KEYS).issubset(provenance)

    first_row = payload["rows"][0]
    assert set(REQUIRED_PROBE_ROW_FIELDNAMES).issubset(first_row)
    assert first_row["layer_group"] == first_row["layer_group_id"]
    assert first_row["timestep_band"] == first_row["timestep_band_id"]
    assert first_row["utility"] == first_row["utility_score"]

    with (output_dir / "cell_utility.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == UTILITY_RECORD_FIELDNAMES
        rows = list(reader)
    assert len(rows) == 4
