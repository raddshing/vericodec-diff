from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vericodec_diff.patch_metrics import (
    PATCH_COUNT,
    SampleRecord,
    build_patch_error_payload,
    write_patch_error_npz,
)
from vericodec_diff.verifier_features import (
    DEFAULT_SIGNAL_NAMES,
    VerifierSampleRecord,
    build_feature_payload,
    write_feature_npz,
)
from vericodec_diff.verifier_training import (
    BUDGET_CURVES_FILENAME,
    CHECKPOINT_INDEX_FILENAME,
    EVAL_JSON_FILENAME,
)


TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_verifier.py"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_verifier.py"
MEMO_SCRIPT = REPO_ROOT / "scripts" / "write_kill_memo.py"


def _make_signal_arrays(offset: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    indices = np.arange(PATCH_COUNT, dtype=np.float32)
    group_a = ((indices.astype(np.int32) + offset) % 16) == 0
    group_b = ((indices.astype(np.int32) + offset) % 16) == 8
    labels = np.logical_or(group_a, group_b)

    signal_a = np.where(group_a, 2.5, 0.05).astype(np.float32)
    signal_a += 0.02 * np.sin(indices * 0.31 + float(offset)).astype(np.float32)

    signal_b = np.where(group_b, 2.5, 0.05).astype(np.float32)
    signal_b += 0.02 * np.cos(indices * 0.27 + float(offset)).astype(np.float32)

    signal_arrays = {
        DEFAULT_SIGNAL_NAMES[0]: signal_a,
        DEFAULT_SIGNAL_NAMES[1]: signal_b,
        DEFAULT_SIGNAL_NAMES[2]: (0.10 + 0.08 * np.sin(indices * 0.11 + float(offset))).astype(np.float32),
        DEFAULT_SIGNAL_NAMES[3]: (2.80 - signal_a + 0.03 * np.cos(indices * 0.07)).astype(np.float32),
        DEFAULT_SIGNAL_NAMES[4]: (2.80 - signal_b + 0.03 * np.sin(indices * 0.09)).astype(np.float32),
        DEFAULT_SIGNAL_NAMES[5]: (0.05 + (indices % 9.0) / 9.0).astype(np.float32),
    }

    metric_values = np.where(labels, 0.12, 0.01).astype(np.float32)
    metric_values[group_a] += np.float32(0.02)
    return signal_arrays, metric_values


def _write_feature_and_error_artifacts(repo_root: Path, *, split: str, sample_id: str, offset: int) -> None:
    signal_arrays, metric_values = _make_signal_arrays(offset)

    feature_record = VerifierSampleRecord(
        sample_id=sample_id,
        split=split,
        image_path=repo_root / "data" / "images" / f"{sample_id}.png",
        trace_path=None,
        trace_status="missing",
        image_local_path=f"data/images/{sample_id}.png",
        trace_local_path="",
        metadata={},
    )
    feature_payload = build_feature_payload(
        feature_record,
        signal_arrays=signal_arrays,
        missing_signals={},
        config={
            "paths": {"feature_root": str(repo_root / "data" / "processed" / "verifier_features")},
            "signals": {"enabled": list(DEFAULT_SIGNAL_NAMES)},
            "trace": {"wavelet": "haar"},
        },
        decode_reencode_backend_name="synthetic",
    )
    write_feature_npz(
        repo_root / "data" / "processed" / "verifier_features" / split / f"{sample_id}__signals.npz",
        feature_payload,
    )

    error_record = SampleRecord(
        sample_id=sample_id,
        split=split,
        source_path=repo_root / "data" / "source" / f"{sample_id}.png",
        reconstruction_path=repo_root / "data" / "recon" / f"{sample_id}.png",
        source_local_path=f"data/source/{sample_id}.png",
        reconstruction_local_path=f"data/recon/{sample_id}.png",
        metadata={},
    )
    error_payload = build_patch_error_payload(
        error_record,
        {
            "lpips": metric_values,
            "one_minus_ssim": (metric_values * 0.5).astype(np.float32),
            "hf_wavelet_l1": (metric_values * 1.5).astype(np.float32),
        },
        lpips_backend="pixel_l2_debug",
        wavelet="haar",
    )
    write_patch_error_npz(
        repo_root / "outputs" / "error_maps" / split / f"{sample_id}__patch64.npz",
        error_payload,
    )


def _build_synthetic_repo(repo_root: Path) -> None:
    _write_feature_and_error_artifacts(repo_root, split="train", sample_id="train_a", offset=0)
    _write_feature_and_error_artifacts(repo_root, split="train", sample_id="train_b", offset=3)
    _write_feature_and_error_artifacts(repo_root, split="val", sample_id="val_a", offset=5)
    _write_feature_and_error_artifacts(repo_root, split="kill", sample_id="kill_a", offset=7)


class VerifierTrainingTests(unittest.TestCase):
    def test_verifier_scripts_write_checkpoints_eval_and_kill_memo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            _build_synthetic_repo(repo_root)

            train_command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                "--repo-root",
                tmpdir,
                "--models-enabled",
                "logistic,mlp",
                "--models-seed",
                "11",
                "--logistic-learning-rate",
                "0.2",
                "--logistic-epochs",
                "120",
                "--mlp-hidden-dim",
                "8",
                "--mlp-learning-rate",
                "0.02",
                "--mlp-epochs",
                "100",
            ]
            subprocess.run(train_command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            eval_command = [
                sys.executable,
                str(EVAL_SCRIPT),
                "--repo-root",
                tmpdir,
            ]
            subprocess.run(eval_command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            memo_command = [
                sys.executable,
                str(MEMO_SCRIPT),
                "--repo-root",
                tmpdir,
            ]
            subprocess.run(memo_command, cwd=REPO_ROOT, check=True, capture_output=True, text=True)

            checkpoint_dir = repo_root / "checkpoints" / "verifier"
            checkpoint_index_path = checkpoint_dir / CHECKPOINT_INDEX_FILENAME
            training_summary_path = checkpoint_dir / "training_summary__patch64.json"
            eval_output_dir = repo_root / "outputs" / "metrics" / "kill_test"
            eval_json_path = eval_output_dir / EVAL_JSON_FILENAME
            budget_csv_path = eval_output_dir / BUDGET_CURVES_FILENAME
            memo_md_path = repo_root / "outputs" / "memos" / "kill_test_memo.md"
            memo_json_path = repo_root / "outputs" / "memos" / "kill_test_memo.json"

            self.assertTrue((checkpoint_dir / "resolved_config.yaml").is_file())
            self.assertTrue(checkpoint_index_path.is_file())
            self.assertTrue(training_summary_path.is_file())
            self.assertTrue((eval_output_dir / "resolved_config.yaml").is_file())
            self.assertTrue(eval_json_path.is_file())
            self.assertTrue(budget_csv_path.is_file())
            self.assertTrue((repo_root / "outputs" / "memos" / "resolved_config.yaml").is_file())
            self.assertTrue(memo_md_path.is_file())
            self.assertTrue(memo_json_path.is_file())

            with checkpoint_index_path.open("r", encoding="utf-8") as handle:
                checkpoint_index = json.load(handle)
            self.assertEqual(len(checkpoint_index["checkpoints"]), 2)
            self.assertIn(
                checkpoint_index["best_checkpoint"]["model_type"],
                {"logistic", "mlp"},
            )

            with eval_json_path.open("r", encoding="utf-8") as handle:
                evaluation = json.load(handle)
            self.assertGreater(evaluation["best_model"]["test_metrics"]["auprc"], 0.95)
            self.assertGreater(
                evaluation["best_model"]["test_metrics"]["auprc"],
                evaluation["heuristic_baseline"]["test_metrics"]["auprc"],
            )
            self.assertEqual(evaluation["artifacts"]["budget_curves_csv"], "outputs/metrics/kill_test/verifier_budget_curves.csv")

            with budget_csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertEqual({row["curve_role"] for row in rows}, {"best_model", "heuristic_baseline"})

            with memo_json_path.open("r", encoding="utf-8") as handle:
                memo = json.load(handle)
            self.assertEqual(memo["decision"], "GO")
            self.assertTrue(all(check["passed"] for check in memo["checks"]))

            memo_text = memo_md_path.read_text(encoding="utf-8")
            self.assertIn("Decision: **GO**", memo_text)
            self.assertIn("## Gate Checks", memo_text)

    def test_eval_script_saves_resolved_config_before_missing_checkpoint_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            command = [
                sys.executable,
                str(EVAL_SCRIPT),
                "--repo-root",
                tmpdir,
            ]
            completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)

            self.assertNotEqual(completed.returncode, 0)
            self.assertTrue((repo_root / "outputs" / "metrics" / "kill_test" / "resolved_config.yaml").is_file())
            self.assertIn("run scripts/train_verifier.py first", completed.stderr)


if __name__ == "__main__":
    unittest.main()
