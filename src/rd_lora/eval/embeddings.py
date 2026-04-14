from __future__ import annotations

from pathlib import Path
from typing import Sequence, TypeVar

import numpy as np


T = TypeVar("T")

_IMAGE_BATCH_SIZE = 8
_TEXT_BATCH_SIZE = 32


def _batched(items: Sequence[T], batch_size: int) -> list[Sequence[T]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _empty_embeddings() -> np.ndarray:
    return np.empty((0, 0), dtype=np.float32)


def cosine_to_unit_interval(values: np.ndarray) -> np.ndarray:
    """
    Normalize cosine similarity from [-1, 1] to [0, 1].
    Formula: clip((values + 1) / 2, 0, 1)
    """
    values = np.asarray(values, dtype=np.float32)
    return np.clip((values + 1.0) / 2.0, 0.0, 1.0)


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
    query = np.asarray(query, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)

    if query.ndim != 2 or ref.ndim != 2:
        raise ValueError("query and ref must be rank-2 arrays")
    if query.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if ref.shape[0] == 0:
        return np.full((query.shape[0],), -1.0, dtype=np.float32)
    if query.shape[1] != ref.shape[1]:
        raise ValueError("query and ref must have the same embedding dimension")

    similarities = query @ ref.T
    return similarities.max(axis=1).astype(np.float32, copy=False)


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
    image_paths = list(image_paths)
    if not image_paths:
        return _empty_embeddings()

    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    encoded_batches: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for batch_paths in _batched(image_paths, _IMAGE_BATCH_SIZE):
                images = []
                for image_path in batch_paths:
                    with Image.open(image_path) as image:
                        images.append(image.convert("RGB"))
                inputs = processor(images=images, return_tensors="pt")
                inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
                features = model.get_image_features(**inputs)
                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output if hasattr(features, 'pooler_output') else features[0]
                features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
                encoded_batches.append(features.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(encoded_batches, axis=0)
    finally:
        del model
        del processor
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


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
    texts = list(texts)
    if not texts:
        return _empty_embeddings()

    import torch
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    encoded_batches: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for batch_texts in _batched(texts, _TEXT_BATCH_SIZE):
                inputs = processor(
                    text=list(batch_texts),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
                features = model.get_text_features(**inputs)
                if not isinstance(features, torch.Tensor):
                    features = features.pooler_output if hasattr(features, 'pooler_output') else features[0]
                features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
                encoded_batches.append(features.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(encoded_batches, axis=0)
    finally:
        del model
        del processor
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


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
    image_paths = list(image_paths)
    if not image_paths:
        return _empty_embeddings()

    import torch
    from PIL import Image
    from transformers import ViTImageProcessor, ViTModel

    model = ViTModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = model.to(device=device, dtype=torch.float32)
    model.eval()

    encoded_batches: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for batch_paths in _batched(image_paths, _IMAGE_BATCH_SIZE):
                images = []
                for image_path in batch_paths:
                    with Image.open(image_path) as image:
                        images.append(image.convert("RGB"))
                inputs = processor(images=images, return_tensors="pt")
                inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
                outputs = model(**inputs)
                features = outputs.last_hidden_state[:, 0, :]
                features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
                encoded_batches.append(features.cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(encoded_batches, axis=0)
    finally:
        del model
        del processor
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
