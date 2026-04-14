Branch: pivot/rd-lora-diff
Module: M05.3b — Task 3: embedding and OCR metric primitives

## Objective
Create low-level primitives for CLIP image/text encoding, DINO image encoding,
cosine similarity utilities, and OCR with character error rate / exact match.
These are building blocks for Task 4 metric calculators.

## Files to create
1. src/rd_lora/eval/embeddings.py
2. src/rd_lora/eval/ocr.py

Do NOT edit any existing files.

## src/rd_lora/eval/embeddings.py

All functions operate on batches. Use transformers library for models.

### encode_images_clip

```python
from pathlib import Path
from typing import Sequence
import numpy as np

def encode_images_clip(
    image_paths: Sequence[Path],
    device: str = "cuda:0",
    model_name: str = "openai/clip-vit-large-patch14",
) -> np.ndarray:
    """
    Encode images into CLIP image embeddings.
    
    Steps:
    1. Load CLIPModel and CLIPProcessor from transformers
    2. For each image path, open with PIL.Image
    3. Process through CLIPProcessor
    4. Get image features via model.get_image_features()
    5. L2-normalize embeddings
    6. Return numpy array of shape (N, D)
    
    Process in batches of 8 to manage VRAM.
    Move model to device, use torch.no_grad().
    Delete model after use to free VRAM.
    """
```

### encode_texts_clip

```python
def encode_texts_clip(
    texts: Sequence[str],
    device: str = "cuda:0",
    model_name: str = "openai/clip-vit-large-patch14",
) -> np.ndarray:
    """
    Encode text strings into CLIP text embeddings.
    
    Steps:
    1. Load CLIPModel and CLIPProcessor from transformers
    2. Process texts through CLIPProcessor
    3. Get text features via model.get_text_features()
    4. L2-normalize embeddings
    5. Return numpy array of shape (N, D)
    
    Process in batches of 32.
    Move model to device, use torch.no_grad().
    Delete model after use to free VRAM.
    """
```

### encode_images_dino

```python
def encode_images_dino(
    image_paths: Sequence[Path],
    device: str = "cuda:0",
    model_name: str = "facebook/dino-vitb16",
) -> np.ndarray:
    """
    Encode images into DINO embeddings.
    
    Steps:
    1. Load ViTModel and ViTImageProcessor from transformers
    2. For each image path, open with PIL.Image, convert to RGB
    3. Process through ViTImageProcessor
    4. Get CLS token output from model (last_hidden_state[:, 0, :])
    5. L2-normalize embeddings
    6. Return numpy array of shape (N, D)
    
    Process in batches of 8 to manage VRAM.
    Move model to device, use torch.no_grad().
    Delete model after use to free VRAM.
    """
```

### cosine_to_unit_interval

```python
def cosine_to_unit_interval(values: np.ndarray) -> np.ndarray:
    """
    Normalize cosine similarity from [-1, 1] to [0, 1].
    Formula: clip((values + 1) / 2, 0, 1)
    """
```

### rowwise_max_cosine

```python
def rowwise_max_cosine(
    query: np.ndarray,
    ref: np.ndarray,
) -> np.ndarray:
    """
    For each query vector, compute max cosine similarity to any ref vector.
    
    query: shape (Q, D)
    ref: shape (R, D)
    Returns: shape (Q,) — max cosine similarity per query row
    
    Both inputs must be L2-normalized already.
    Compute via matrix multiply: sim = query @ ref.T, then max over axis=1.
    """
```

Implementation notes:
- All embedding functions should handle empty input gracefully (return empty array)
- Use float32 for model inference even if torch_dtype is float16
- Always L2-normalize before returning
- Import torch only inside functions to avoid import-time GPU allocation
- Each encode function should load and delete the model within the function call
  to avoid holding VRAM between calls

## src/rd_lora/eval/ocr.py

### run_ocr

```python
from pathlib import Path
from typing import Sequence

def run_ocr(
    image_paths: Sequence[Path],
    engine: str = "tesseract",
) -> list[str]:
    """
    Run OCR on a list of images and return extracted text.
    
    Supported engines:
    - "tesseract": use pytesseract.image_to_string()
    - "trocr": use TrOCRProcessor + VisionEncoderDecoderModel from transformers
    
    Steps:
    1. For each image, open with PIL.Image
    2. Run OCR with the selected engine
    3. Strip whitespace from output
    4. If OCR returns empty string, keep as empty string
    5. Return list of extracted text strings
    
    For tesseract:
    - import pytesseract
    - use pytesseract.image_to_string(image).strip()
    - if pytesseract not importable, raise ImportError with helpful message
    
    For trocr:
    - use "microsoft/trocr-base-printed"
    - process in batches of 8
    - delete model after use
    
    Default to tesseract for simplicity and availability.
    """
```

### character_error_rate

```python
def character_error_rate(pred: str, target: str) -> float:
    """
    Compute character-level edit distance (Levenshtein) normalized by target length.
    
    If target is empty and pred is empty, return 0.0.
    If target is empty and pred is non-empty, return 1.0.
    If pred is empty and target is non-empty, return 1.0.
    
    Use dynamic programming for edit distance.
    CER = edit_distance(pred, target) / max(len(target), 1)
    Clamp to [0.0, 1.0] range.
    """
```

### exact_match

```python
def exact_match(pred: str, target: str) -> int:
    """
    Return 1 if pred.strip().upper() == target.strip().upper(), else 0.
    Case-insensitive comparison after stripping whitespace.
    """
```

Implementation notes:
- character_error_rate should implement Levenshtein distance directly
  (do NOT require python-Levenshtein or editdistance packages)
- Use standard library only for CER computation
- For tesseract, handle the case where pytesseract is not installed
  by raising a clear ImportError

## Tests

Do NOT create test files for Task 3. Tests for these primitives will be
included in Task 4 test files (test_rdlora_eval_metrics_subject.py and
test_rdlora_eval_metrics_style.py) which test the full metric pipeline
including these primitives.

## Constraints
- Do NOT edit any existing files
- Do NOT import from adapter_factory, execution, or timestep_routing
- Use transformers for CLIP and DINO models
- Use pytesseract for tesseract OCR
- Implement Levenshtein distance without external packages
- All functions must handle empty inputs gracefully
- No GPU allocation at import time

## Validation
After completion:
    python3 -c "import ast; ast.parse(open('src/rd_lora/eval/embeddings.py').read()); print('embeddings OK')"
    python3 -c "import ast; ast.parse(open('src/rd_lora/eval/ocr.py').read()); print('ocr OK')"
    PYTHONPATH=src python3 -c "from rd_lora.eval.embeddings import encode_images_clip, encode_texts_clip, encode_images_dino, cosine_to_unit_interval, rowwise_max_cosine; print('embeddings import OK')"
    PYTHONPATH=src python3 -c "from rd_lora.eval.ocr import run_ocr, character_error_rate, exact_match; print('ocr import OK')"
    PYTHONPATH=src python3 -c "from rd_lora.eval.ocr import character_error_rate, exact_match; assert character_error_rate('HELLO', 'HELLO') == 0.0; assert character_error_rate('', 'HELLO') == 1.0; assert exact_match('hello', 'HELLO') == 1; assert exact_match('helo', 'HELLO') == 0; print('ocr unit OK')"
    PYTHONPATH=src python3 -c "import numpy as np; from rd_lora.eval.embeddings import cosine_to_unit_interval, rowwise_max_cosine; v = np.array([1.0, -1.0, 0.0]); r = cosine_to_unit_interval(v); assert abs(r[0]-1.0)<1e-6 and abs(r[1]-0.0)<1e-6 and abs(r[2]-0.5)<1e-6; print('cosine OK')"
Expected: all OK, no errors.
