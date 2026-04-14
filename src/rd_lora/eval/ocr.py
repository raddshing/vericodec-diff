from __future__ import annotations

from pathlib import Path
from typing import Sequence, TypeVar


T = TypeVar("T")

_OCR_BATCH_SIZE = 8


def _batched(items: Sequence[T], batch_size: int) -> list[Sequence[T]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


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
    image_paths = list(image_paths)
    if not image_paths:
        return []

    normalized_engine = engine.lower()

    if normalized_engine == "tesseract":
        try:
            import pytesseract
        except ImportError as exc:
            raise ImportError("pytesseract is required for engine='tesseract'") from exc

        from PIL import Image

        outputs: list[str] = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                outputs.append(pytesseract.image_to_string(image).strip())
        return outputs

    if normalized_engine == "trocr":
        import torch
        from PIL import Image
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_name = "microsoft/trocr-base-printed"
        processor = TrOCRProcessor.from_pretrained(model_name)
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        model = model.to(device=device, dtype=torch.float32)
        model.eval()

        outputs: list[str] = []
        try:
            with torch.no_grad():
                for batch_paths in _batched(image_paths, _OCR_BATCH_SIZE):
                    images = []
                    for image_path in batch_paths:
                        with Image.open(image_path) as image:
                            images.append(image.convert("RGB"))
                    pixel_values = processor(images=images, return_tensors="pt").pixel_values
                    pixel_values = pixel_values.to(device=device, dtype=torch.float32)
                    generated_ids = model.generate(pixel_values)
                    outputs.extend(text.strip() for text in processor.batch_decode(generated_ids, skip_special_tokens=True))
            return outputs
        finally:
            del model
            del processor
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise ValueError(f"Unsupported OCR engine: {engine}")


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
    if not target and not pred:
        return 0.0
    if not target or not pred:
        return 1.0

    previous_row = list(range(len(target) + 1))
    for pred_index, pred_char in enumerate(pred, start=1):
        current_row = [pred_index]
        for target_index, target_char in enumerate(target, start=1):
            insertion_cost = current_row[target_index - 1] + 1
            deletion_cost = previous_row[target_index] + 1
            substitution_cost = previous_row[target_index - 1] + (pred_char != target_char)
            current_row.append(min(insertion_cost, deletion_cost, substitution_cost))
        previous_row = current_row

    normalized_distance = previous_row[-1] / max(len(target), 1)
    return max(0.0, min(1.0, float(normalized_distance)))


def exact_match(pred: str, target: str) -> int:
    """
    Return 1 if pred.strip().upper() == target.strip().upper(), else 0.
    Case-insensitive comparison after stripping whitespace.
    """
    return int(pred.strip().upper() == target.strip().upper())
