from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBJECT_PATH = REPO_ROOT / "data" / "rd_lora" / "eval" / "subject_personalization_eval.csv"
STYLE_PATH = REPO_ROOT / "data" / "rd_lora" / "eval" / "style_domain_eval.csv"
VALIDATE_SCRIPT = REPO_ROOT / "scripts" / "validate_rdlora_eval_manifests.py"
REQUIRED_COLUMNS = ("prompt_id", "task", "prompt", "seed", "expected_text")
SUBJECT_PROMPT_IDS = tuple(f"subj_{index:02d}" for index in range(1, 9))
STYLE_PROMPT_IDS = tuple(f"sign_{index:02d}" for index in range(1, 17))
SUBJECT_SEEDS = (42, 123, 456, 789)
STYLE_SEEDS = (42, 123)


def _read_manifest(path: Path) -> pd.DataFrame:
    assert path.is_file(), f"Missing manifest: {path}"
    frame = pd.read_csv(path, keep_default_na=False)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    assert not missing_columns, f"{path.name} is missing required columns: {missing_columns}"
    normalized = frame.loc[:, REQUIRED_COLUMNS].copy()
    normalized["seed"] = pd.to_numeric(normalized["seed"], errors="raise").astype(int)
    for column in ("prompt_id", "task", "prompt", "expected_text"):
        normalized[column] = normalized[column].astype(str)
    return normalized.reset_index(drop=True)


def _assert_sorted(frame: pd.DataFrame) -> None:
    sorted_frame = frame.sort_values(["prompt_id", "seed"], kind="stable").reset_index(drop=True)
    assert frame.equals(sorted_frame)


def _assert_seed_grid(frame: pd.DataFrame, *, prompt_ids: tuple[str, ...], seeds: tuple[int, ...]) -> None:
    assert tuple(sorted(frame["prompt_id"].unique())) == prompt_ids
    assert tuple(sorted(set(frame["seed"]))) == seeds
    assert not frame.duplicated(subset=["prompt_id", "seed"]).any()

    expected_seeds = tuple(seeds)
    for prompt_id, seed_series in frame.groupby("prompt_id", sort=True)["seed"]:
        assert tuple(sorted(seed_series.tolist())) == expected_seeds, prompt_id


def test_eval_manifest_files_exist_and_parse_without_error() -> None:
    for path in (SUBJECT_PATH, STYLE_PATH):
        frame = _read_manifest(path)
        assert tuple(frame.columns) == REQUIRED_COLUMNS
        assert not frame.empty


def test_subject_personalization_manifest_matches_contract() -> None:
    frame = _read_manifest(SUBJECT_PATH)

    assert len(frame) == 32
    assert set(frame["task"]) == {"subject_personalization"}
    assert (frame["expected_text"] == "").all()
    assert frame["prompt"].str.contains("sks dog", regex=False).all()
    _assert_seed_grid(frame, prompt_ids=SUBJECT_PROMPT_IDS, seeds=SUBJECT_SEEDS)
    _assert_sorted(frame)


def test_style_domain_manifest_matches_contract() -> None:
    frame = _read_manifest(STYLE_PATH)

    assert len(frame) == 32
    assert set(frame["task"]) == {"style_domain"}
    assert (frame["expected_text"].str.strip() != "").all()
    assert frame.apply(lambda row: f'"{row["expected_text"]}"' in row["prompt"], axis=1).all()
    _assert_seed_grid(frame, prompt_ids=STYLE_PROMPT_IDS, seeds=STYLE_SEEDS)
    _assert_sorted(frame)


def test_validator_script_accepts_committed_manifests(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(VALIDATE_SCRIPT),
            "--subject",
            str(SUBJECT_PATH),
            "--style",
            str(STYLE_PATH),
            "--output_dir",
            str(tmp_path / "validation"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "validation" / "resolved_config.yaml").is_file()
    assert (tmp_path / "validation" / "validation_summary.json").is_file()
