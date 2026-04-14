from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rd_lora.eval.embeddings import (
    cosine_to_unit_interval,
    encode_images_clip,
    encode_texts_clip,
)
from rd_lora.eval.ocr import character_error_rate, exact_match, run_ocr
from rd_lora.eval.score import style_score


def _normalize_expected_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    if text.strip() == "":
        return ""
    return text


def compute_style_metrics(
    generated_records_csv: Path,
    device: str = "cuda:0",
    ocr_engine: str = "tesseract",
) -> dict[str, float]:
    """
    Compute style_domain metrics from generated signage images.

    Steps:
    1. Read generated_records_csv with pandas
    2. Collect generated image paths and expected_text from each row
    3. Fail closed if no generated images or expected_text column missing

    4. Run OCR on all generated images -> list of predicted text strings

    5. For each image, compute:
       a. exact_match(pred, expected) -> 0 or 1
       b. character_error_rate(pred, expected) -> float in [0,1]

    6. ocr_exact_match = mean of all exact_match values (fraction correct)
    7. ocr_cer_mean = mean of all CER values

    8. Encode generated images with CLIP image encoder
    9. Encode prompts from "prompt" column with CLIP text encoder
    10. Compute row-wise cosine similarity, normalize to [0,1]
    11. clip_t_mean = mean of normalized similarities

    12. Compute composite score via style_score()

    Returns dict with:
      ocr_exact_match: float
      ocr_cer_mean: float
      ocr_text_score: float  (0.70 * (1 - cer) + 0.30 * em, for reference)
      clip_t_mean: float
      score: float
      num_images: int
    """
    frame = pd.read_csv(generated_records_csv)
    if frame.empty:
        raise ValueError(f"generated_records_csv is empty: {generated_records_csv}")

    required_columns = {"image_path", "prompt", "expected_text"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"generated_records_csv is missing required columns: {missing_columns}")

    sorted_frame = frame.sort_values("image_path", kind="stable").reset_index(drop=True)
    image_paths = [Path(str(value)) for value in sorted_frame["image_path"].tolist()]
    prompts = ["" if pd.isna(value) else str(value) for value in sorted_frame["prompt"].tolist()]
    expected_texts = [_normalize_expected_text(value) for value in sorted_frame["expected_text"].tolist()]

    if not image_paths:
        raise ValueError("No generated images found in generated_records_csv")

    predicted_texts = run_ocr(image_paths, engine=ocr_engine)

    exact_matches: list[float] = []
    cer_values: list[float] = []
    for predicted_text, expected_text in zip(predicted_texts, expected_texts, strict=True):
        if expected_text == "":
            exact_matches.append(0.0)
            cer_values.append(1.0)
            continue
        exact_matches.append(float(exact_match(predicted_text, expected_text)))
        cer_values.append(float(character_error_rate(predicted_text, expected_text)))

    ocr_exact_match = round(float(np.mean(exact_matches, dtype=np.float64)), 6)
    ocr_cer_mean = round(float(np.mean(cer_values, dtype=np.float64)), 6)

    gen_clip_i = encode_images_clip(image_paths, device=device)
    text_emb = encode_texts_clip(prompts, device=device)
    clip_t = np.sum(gen_clip_i * text_emb, axis=1, dtype=np.float32)
    clip_t_mean = round(float(np.mean(cosine_to_unit_interval(clip_t), dtype=np.float64)), 6)

    ocr_cer_score = max(0.0, 1.0 - min(1.0, ocr_cer_mean))
    ocr_text_score = round(float(0.70 * ocr_cer_score + 0.30 * ocr_exact_match), 6)

    result: dict[str, Any] = {
        "ocr_exact_match": ocr_exact_match,
        "ocr_cer_mean": ocr_cer_mean,
        "ocr_text_score": ocr_text_score,
        "clip_t_mean": clip_t_mean,
        "score": style_score(
            ocr_exact_match=ocr_exact_match,
            ocr_cer_mean=ocr_cer_mean,
            clip_t_mean=clip_t_mean,
        ),
        "num_images": len(image_paths),
    }
    return result
