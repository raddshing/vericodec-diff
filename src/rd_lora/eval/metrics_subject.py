from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rd_lora.eval.embeddings import (
    cosine_to_unit_interval,
    encode_images_clip,
    encode_images_dino,
    encode_texts_clip,
    rowwise_max_cosine,
)
from rd_lora.eval.score import subject_score


def _list_reference_images(reference_image_dir: Path) -> list[Path]:
    reference_paths: list[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.ppm"):
        reference_paths.extend(path for path in reference_image_dir.glob(pattern) if path.is_file())
    return sorted(reference_paths)


def compute_subject_metrics(
    generated_records_csv: Path,
    reference_image_dir: Path,
    device: str = "cuda:0",
) -> dict[str, float]:
    """
    Compute subject_personalization metrics from generated images.

    Steps:
    1. Read generated_records_csv with pandas
    2. Collect all generated image paths from the "image_path" column
    3. Collect all reference images from reference_image_dir (*.png, *.jpg, *.jpeg)
    4. Fail closed if no generated images or no reference images

    5. Encode generated images with DINO -> gen_dino (N, D)
    6. Encode reference images with DINO -> ref_dino (R, D)
    7. For each generated image, compute max cosine sim to any ref image in DINO space
    8. Normalize to [0,1] via cosine_to_unit_interval
    9. dino_ref_max_mean = mean of normalized max similarities

    10. Encode generated images with CLIP image encoder -> gen_clip_i (N, D)
    11. Encode reference images with CLIP image encoder -> ref_clip_i (R, D)
    12. For each generated image, compute max cosine sim to any ref image in CLIP-I space
    13. Normalize to [0,1]
    14. clip_i_ref_max_mean = mean of normalized max similarities

    15. Encode prompts from "prompt" column with CLIP text encoder -> text_emb (N, D)
    16. Compute per-image cosine similarity between gen_clip_i and text_emb (row-wise dot product)
    17. Normalize to [0,1]
    18. clip_t_mean = mean of normalized similarities

    19. Compute composite score via subject_score()

    Returns dict with:
      dino_ref_max_mean: float
      clip_i_ref_max_mean: float
      clip_t_mean: float
      score: float
      num_images: int
    """
    frame = pd.read_csv(generated_records_csv)
    if frame.empty:
        raise ValueError(f"generated_records_csv is empty: {generated_records_csv}")

    required_columns = {"image_path", "prompt"}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"generated_records_csv is missing required columns: {missing_columns}")

    sorted_frame = frame.sort_values("image_path", kind="stable").reset_index(drop=True)
    generated_image_paths = [Path(str(value)) for value in sorted_frame["image_path"].tolist()]
    prompts = ["" if pd.isna(value) else str(value) for value in sorted_frame["prompt"].tolist()]
    reference_image_paths = _list_reference_images(reference_image_dir)

    if not generated_image_paths:
        raise ValueError("No generated images found in generated_records_csv")
    if not reference_image_paths:
        raise ValueError(f"No reference images found in {reference_image_dir}")

    gen_dino = encode_images_dino(generated_image_paths, device=device)
    ref_dino = encode_images_dino(reference_image_paths, device=device)
    dino_ref_max = cosine_to_unit_interval(rowwise_max_cosine(gen_dino, ref_dino))
    dino_ref_max_mean = round(float(np.mean(dino_ref_max, dtype=np.float64)), 6)

    gen_clip_i = encode_images_clip(generated_image_paths, device=device)
    ref_clip_i = encode_images_clip(reference_image_paths, device=device)
    clip_i_ref_max = cosine_to_unit_interval(rowwise_max_cosine(gen_clip_i, ref_clip_i))
    clip_i_ref_max_mean = round(float(np.mean(clip_i_ref_max, dtype=np.float64)), 6)

    text_emb = encode_texts_clip(prompts, device=device)
    clip_t = np.sum(gen_clip_i * text_emb, axis=1, dtype=np.float32)
    clip_t_mean = round(float(np.mean(cosine_to_unit_interval(clip_t), dtype=np.float64)), 6)

    result: dict[str, Any] = {
        "dino_ref_max_mean": dino_ref_max_mean,
        "clip_i_ref_max_mean": clip_i_ref_max_mean,
        "clip_t_mean": clip_t_mean,
        "score": subject_score(
            dino_ref_max_mean=dino_ref_max_mean,
            clip_i_ref_max_mean=clip_i_ref_max_mean,
            clip_t_mean=clip_t_mean,
        ),
        "num_images": len(generated_image_paths),
    }
    return result
