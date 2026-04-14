from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.eval import metrics_style
from rd_lora.eval.score import style_score


def _normalized_embeddings(rows: int, dims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(rows, dims)).astype(np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / norms


def _write_dummy_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1, 1), color=(255, 255, 255)).save(path)


def test_style_score_known_inputs() -> None:
    assert style_score(0.5, 0.2, 0.8) == 0.737


def test_style_score_perfect_ocr_returns_one() -> None:
    assert style_score(1.0, 0.0, 1.0) == 1.0


def test_style_score_worst_ocr_returns_zero() -> None:
    assert style_score(0.0, 1.0, 0.0) == 0.0


def test_compute_style_metrics_with_mocked_ocr_and_embeddings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    generated_dir = tmp_path / "generated"
    generated_rows: list[dict[str, str]] = []
    for index, expected_text in enumerate(["HELLO", "LEFT", "SOUP"]):
        image_path = generated_dir / f"style_{index:02d}.png"
        _write_dummy_png(image_path)
        generated_rows.append(
            {
                "image_path": str(image_path),
                "prompt": f'sign that says "{expected_text}"',
                "expected_text": expected_text,
            }
        )
    generated_records_csv = tmp_path / "generated_records.csv"
    pd.DataFrame(generated_rows).to_csv(generated_records_csv, index=False)

    monkeypatch.setattr(metrics_style, "run_ocr", lambda image_paths, engine="tesseract": ["HELLO", "L3FT", ""])
    monkeypatch.setattr(
        metrics_style,
        "encode_images_clip",
        lambda image_paths, device="cuda:0": _normalized_embeddings(len(image_paths), 4, seed=41 + len(image_paths)),
    )
    monkeypatch.setattr(
        metrics_style,
        "encode_texts_clip",
        lambda texts, device="cuda:0": _normalized_embeddings(len(texts), 4, seed=51 + len(texts)),
    )

    result = metrics_style.compute_style_metrics(
        generated_records_csv=generated_records_csv,
        device="cpu",
        ocr_engine="tesseract",
    )

    assert set(result) == {
        "ocr_exact_match",
        "ocr_cer_mean",
        "ocr_text_score",
        "clip_t_mean",
        "score",
        "num_images",
    }
    assert result["num_images"] == 3
    assert 0.0 <= result["ocr_exact_match"] <= 1.0
    assert 0.0 <= result["ocr_cer_mean"] <= 1.0
    assert 0.0 <= result["ocr_text_score"] <= 1.0
    assert 0.0 <= result["clip_t_mean"] <= 1.0
    assert 0.0 <= result["score"] <= 1.0
