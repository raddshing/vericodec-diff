Branch: pivot/rd-lora-diff
Module: M05.3b — Task 4: task-specific metric calculators

## Objective
Create task-specific metric calculators and a score aggregation module.
These use the embedding/OCR primitives from Task 3 to compute the advisor-defined
composite score for each task.

## Files to create
1. src/rd_lora/eval/score.py
2. src/rd_lora/eval/metrics_subject.py
3. src/rd_lora/eval/metrics_style.py
4. tests/test_rdlora_eval_metrics_subject.py
5. tests/test_rdlora_eval_metrics_style.py

Do NOT edit any existing files.

## src/rd_lora/eval/score.py

Exact implementation (advisor-frozen weights):

```python
from __future__ import annotations


def subject_score(
    dino_ref_max_mean: float,
    clip_i_ref_max_mean: float,
    clip_t_mean: float,
) -> float:
    """
    Composite score for subject_personalization task.
    Weights: DINO 0.60, CLIP-I 0.10, CLIP-T 0.30
    All inputs must be in [0, 1] (cosine normalized).
    """
    return round(0.60 * dino_ref_max_mean + 0.10 * clip_i_ref_max_mean + 0.30 * clip_t_mean, 6)


def style_score(
    ocr_exact_match: float,
    ocr_cer_mean: float,
    clip_t_mean: float,
) -> float:
    """
    Composite score for style_domain task.
    ocr_cer_score = max(0, 1 - min(1, ocr_cer_mean))
    ocr_text_score = 0.70 * ocr_cer_score + 0.30 * ocr_exact_match
    score = 0.70 * ocr_text_score + 0.30 * clip_t_mean
    All inputs must be in [0, 1].
    """
    ocr_cer_score = max(0.0, 1.0 - min(1.0, ocr_cer_mean))
    ocr_text_score = 0.70 * ocr_cer_score + 0.30 * ocr_exact_match
    return round(0.70 * ocr_text_score + 0.30 * clip_t_mean, 6)


def backend_score(task_scores: dict[str, float]) -> float:
    """
    Backend-level aggregate: mean of subject and style task scores.
    Requires both "subject_personalization" and "style_domain" keys.
    """
    required = ["subject_personalization", "style_domain"]
    for key in required:
        if key not in task_scores:
            raise ValueError(f"Missing required task score: {key}")
    return round(0.5 * task_scores["subject_personalization"] + 0.5 * task_scores["style_domain"], 6)
```

## src/rd_lora/eval/metrics_subject.py

```python
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
```

Implementation notes:
- reference images: glob reference_image_dir for *.png, *.jpg, *.jpeg
- sort reference images for determinism
- sort generated image paths for determinism
- for clip_t_mean: use encode_images_clip for generated images and encode_texts_clip
  for prompts, then compute row-wise dot product (since both are L2-normalized,
  dot product = cosine similarity)
- all cosine similarities go through cosine_to_unit_interval before averaging
- if generated_records_csv is empty (0 rows), raise ValueError

## src/rd_lora/eval/metrics_style.py

```python
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
```

Implementation notes:
- ocr_text_score is computed as: 0.70 * max(0, 1 - min(1, ocr_cer_mean)) + 0.30 * ocr_exact_match
  (this is an informational field; the composite score uses style_score() directly)
- if expected_text is NaN or empty for any row, treat as empty target -> CER=1.0, EM=0
- sort image paths for determinism

## tests/test_rdlora_eval_metrics_subject.py

Unit test WITHOUT GPU. Use mocking to avoid loading real CLIP/DINO models.

Test:
1. score.subject_score with known inputs produces correct output
   e.g. subject_score(0.8, 0.7, 0.9) == round(0.60*0.8 + 0.10*0.7 + 0.30*0.9, 6)
2. score.subject_score with all zeros returns 0.0
3. score.subject_score with all ones returns 1.0
4. score.backend_score raises ValueError if key missing
5. Mock compute_subject_metrics:
   - Create a fake generated_records_csv with 3 rows
   - Create a fake reference_image_dir with 2 dummy PNG images (1x1 pixel)
   - Mock encode_images_dino to return random normalized vectors
   - Mock encode_images_clip to return random normalized vectors
   - Mock encode_texts_clip to return random normalized vectors
   - Verify output dict has all required keys
   - Verify num_images == 3

## tests/test_rdlora_eval_metrics_style.py

Unit test WITHOUT GPU. Use mocking.

Test:
1. score.style_score with known inputs:
   e.g. style_score(0.5, 0.2, 0.8):
   ocr_cer_score = 1.0 - 0.2 = 0.8
   ocr_text_score = 0.70 * 0.8 + 0.30 * 0.5 = 0.71
   score = 0.70 * 0.71 + 0.30 * 0.8 = 0.737
2. score.style_score with perfect OCR (em=1, cer=0) and clip_t=1.0 returns 1.0
3. score.style_score with worst OCR (em=0, cer=1) and clip_t=0.0 returns 0.0
4. Mock compute_style_metrics:
   - Create a fake generated_records_csv with 3 rows including expected_text
   - Create 3 dummy PNG images (1x1 pixel)
   - Mock run_ocr to return known strings
   - Mock encode_images_clip to return random normalized vectors
   - Mock encode_texts_clip to return random normalized vectors
   - Verify output dict has all required keys
   - Verify num_images == 3

## Constraints
- Do NOT edit any existing files
- Use the exact weight constants from advisor spec (frozen)
- All score functions must round to 6 decimal places
- Tests must NOT require GPU or real model downloads
- Use monkeypatch or unittest.mock for mocking

## Validation
After completion:
    python3 -c "import ast; ast.parse(open('src/rd_lora/eval/score.py').read()); print('score OK')"
    python3 -c "import ast; ast.parse(open('src/rd_lora/eval/metrics_subject.py').read()); print('metrics_subject OK')"
    python3 -c "import ast; ast.parse(open('src/rd_lora/eval/metrics_style.py').read()); print('metrics_style OK')"
    PYTHONPATH=src python3 -c "from rd_lora.eval.score import subject_score, style_score, backend_score; print('score import OK')"
    PYTHONPATH=src python3 -c "from rd_lora.eval.score import subject_score; assert subject_score(1.0, 1.0, 1.0) == 1.0; assert subject_score(0.0, 0.0, 0.0) == 0.0; print('subject_score OK')"
    PYTHONPATH=src python3 -c "from rd_lora.eval.score import style_score; assert style_score(1.0, 0.0, 1.0) == 1.0; assert style_score(0.0, 1.0, 0.0) == 0.0; print('style_score OK')"
    python3 -m pytest tests/test_rdlora_eval_metrics_subject.py tests/test_rdlora_eval_metrics_style.py -x -q 2>&1 | tail -10
    python3 -m pytest tests/test_training_execution_path.py tests/test_noop_band_execution.py tests/test_rdlora_eval_manifests.py tests/test_rdlora_eval_generation.py -x -q 2>&1 | tail -10
Expected: all syntax OK, all imports OK, all score assertions OK, all tests pass.
