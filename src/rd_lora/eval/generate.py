from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - exercised via monkeypatch in unit tests.
    torch = None

try:
    from diffusers import StableDiffusionXLPipeline
except ImportError:  # pragma: no cover - exercised via monkeypatch in unit tests.
    StableDiffusionXLPipeline = None


_GENERATION_RECORD_COLUMNS = [
    "run_dir",
    "backend",
    "task",
    "prompt_id",
    "prompt",
    "seed",
    "expected_text",
    "image_path",
]


def _load_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON artifact not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _resolve_checkpoint_path(run_dir: Path, checkpoint_value: str) -> Path:
    checkpoint_path = Path(checkpoint_value).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (run_dir / checkpoint_path).resolve()
    else:
        checkpoint_path = checkpoint_path.resolve()
    return checkpoint_path


def _resolve_torch_dtype(torch_dtype: str) -> Any:
    if torch is None:
        raise ImportError("torch is required for eval image generation")
    dtype = getattr(torch, torch_dtype, None)
    if dtype is None:
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return dtype


def _extract_image(result: Any) -> Any:
    images = getattr(result, "images", None)
    if isinstance(result, dict):
        images = result.get("images", images)
    if not images:
        raise RuntimeError("Pipeline result did not include any images")
    return images[0]


def load_eval_pipeline(
    base_model_id: str,
    checkpoint_path: Path,
    device: str = "cuda:0",
    torch_dtype: str = "float16",
) -> Any:
    """
    Load the base SDXL pipeline and attach the LoRA checkpoint.

    Steps:
    1. Load StableDiffusionXLPipeline.from_pretrained(base_model_id)
    2. Load LoRA weights from checkpoint_path using pipe.load_lora_weights()
    3. Move to device with correct dtype
    4. Return the pipeline

    Fail closed if checkpoint_path does not exist.
    """
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint directory not found: {checkpoint_path}")
    if StableDiffusionXLPipeline is None:
        raise ImportError("diffusers.StableDiffusionXLPipeline is required for eval image generation")

    dtype = _resolve_torch_dtype(torch_dtype)
    pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=dtype)
    try:
        pipe.load_lora_weights(str(checkpoint_path), weight_name="pytorch_lora_weights.safetensors")
    except TypeError:
        pipe.load_lora_weights(str(checkpoint_path))
    pipe = pipe.to(device=device, torch_dtype=dtype)
    return pipe


def generate_eval_images_for_run(
    run_dir: Path,
    prompt_manifest_path: Path,
    output_dir: Path,
    base_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    device: str = "cuda:0",
    torch_dtype: str = "float16",
) -> Path:
    """
    Generate all evaluation images for one training run.

    Steps:
    1. Read run_dir/train_summary.json to get backend and task
    2. Read run_dir/checkpoint_info.json to get checkpoint_path
    3. Verify checkpoint exists on disk; fail closed if not
    4. Load pipeline via load_eval_pipeline
    5. Read prompt_manifest_path CSV with pandas
    6. For each row in the manifest:
       a. Set generator seed from row["seed"]
       b. Generate one image using pipe(prompt, generator=generator,
          guidance_scale=guidance_scale, num_inference_steps=num_inference_steps)
       c. Save as: output_dir/<backend>__<task>__<prompt_id>__seed<seed>.png
    7. Build generation_records list with columns:
       run_dir, backend, task, prompt_id, prompt, seed, expected_text, image_path
    8. Write generation_records.csv to output_dir
    9. Return path to generation_records.csv

    Rules:
    - Same scheduler/inference_steps/guidance_scale across all backends
    - Deterministic file naming: <backend>__<task>__<prompt_id>__seed<seed>.png
    - Fail closed if checkpoint missing
    - Use torch.Generator(device=device).manual_seed(seed) for reproducibility
    """
    run_dir = Path(run_dir).expanduser().resolve()
    prompt_manifest_path = Path(prompt_manifest_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_summary = _load_json_mapping(run_dir / "train_summary.json")
    checkpoint_info = _load_json_mapping(run_dir / "checkpoint_info.json")

    backend = str(train_summary["backend"])
    task = str(train_summary["task"])
    checkpoint_path = _resolve_checkpoint_path(run_dir, str(checkpoint_info["checkpoint_path"]))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"LoRA checkpoint directory not found: {checkpoint_path}")

    generation_records: list[dict[str, Any]] = []
    pipe: Any | None = None
    try:
        pipe = load_eval_pipeline(
            base_model_id=base_model_id,
            checkpoint_path=checkpoint_path,
            device=device,
            torch_dtype=torch_dtype,
        )

        manifest = pd.read_csv(prompt_manifest_path)
        for row in manifest.to_dict(orient="records"):
            prompt_id = str(row["prompt_id"])
            prompt = str(row["prompt"])
            seed = int(row["seed"])
            expected_text = str(row["expected_text"])
            generator = torch.Generator(device=device).manual_seed(seed)
            result = pipe(
                prompt,
                generator=generator,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )
            image = _extract_image(result)
            image_path = output_dir / f"{backend}__{task}__{prompt_id}__seed{seed}.png"
            image.save(image_path)
            generation_records.append(
                {
                    "run_dir": str(run_dir),
                    "backend": backend,
                    "task": task,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "seed": seed,
                    "expected_text": expected_text,
                    "image_path": str(image_path),
                }
            )

        records_path = output_dir / "generation_records.csv"
        pd.DataFrame(generation_records, columns=_GENERATION_RECORD_COLUMNS).to_csv(records_path, index=False)
        return records_path
    finally:
        if pipe is not None:
            try:
                pipe.unload_lora_weights()
            except Exception:
                # Cleanup should not replace the original generation failure.
                pass
            del pipe
