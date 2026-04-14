from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.eval import metrics_subject
from rd_lora.eval.score import backend_score, subject_score


def _normalized_embeddings(rows: int, dims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, dims)).astype(np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / norms


def _write_dummy_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1, 1), color=(255, 255, 255)).save(path)


def test_subject_score_known_inputs() -> None:
    expected = round(0.60 * 0.8 + 0.10 * 0.7 + 0.30 * 0.9, 6)
    assert subject_score(0.8, 0.7, 0.9) == expected


def test_subject_score_all_zeros() -> None:
    assert subject_score(0.0, 0.0, 0.0) == 0.0


def test_subject_score_all_ones() -> None:
    assert subject_score(1.0, 1.0, 1.0) == 1.0


def test_backend_score_raises_when_task_key_missing() -> None:
    with pytest.raises(ValueError, match="Missing required task score: style_domain"):
        backend_score({"subject_personalization": 0.9})


def test_compute_subject_metrics_with_mocked_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generated_dir = tmp_path / "generated"
    generated_rows: list[dict[str, str]] = []
    for index in range(3):
        image_path = generated_dir / f"gen_{index:02d}.png"
        _write_dummy_png(image_path)
        generated_rows.append(
            {
                "image_path": str(image_path),
                "prompt": f"portrait prompt {index}",
            }
        )
    generated_records_csv = tmp_path / "generated_records.csv"
    pd.DataFrame(generated_rows).to_csv(generated_records_csv, index=False)

    reference_image_dir = tmp_path / "reference"
    for index in range(2):
        _write_dummy_png(reference_image_dir / f"ref_{index:02d}.png")

    def fake_encode_images_dino(image_paths: list[Path], device: str = "cuda:0") -> np.ndarray:
        return _normalized_embeddings(len(image_paths), 4, seed=11 + len(image_paths))

    def fake_encode_images_clip(image_paths: list[Path], device: str = "cuda:0") -> np.ndarray:
        return _normalized_embeddings(len(image_paths), 4, seed=21 + len(image_paths))

    def fake_encode_texts_clip(texts: list[str], device: str = "cuda:0") -> np.ndarray:
        return _normalized_embeddings(len(texts), 4, seed=31 + len(texts))

    monkeypatch.setattr(metrics_subject, "encode_images_dino", fake_encode_images_dino)
    monkeypatch.setattr(metrics_subject, "encode_images_clip", fake_encode_images_clip)
    monkeypatch.setattr(metrics_subject, "encode_texts_clip", fake_encode_texts_clip)

    result = metrics_subject.compute_subject_metrics(
        generated_records_csv=generated_records_csv,
        reference_image_dir=reference_image_dir,
        device="cpu",
    )

    assert set(result) == {
        "dino_ref_max_mean",
        "clip_i_ref_max_mean",
        "clip_t_mean",
        "score",
        "num_images",
    }
    assert result["num_images"] == 3
    assert 0.0 <= result["dino_ref_max_mean"] <= 1.0
    assert 0.0 <= result["clip_i_ref_max_mean"] <= 1.0
    assert 0.0 <= result["clip_t_mean"] <= 1.0
    assert 0.0 <= result["score"] <= 1.0
