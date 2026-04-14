Branch: pivot/rd-lora-diff
Module: M05.3b — Task 2: checkpoint-to-image generation

## Objective
Create a deterministic post-hoc inference pipeline that loads each saved LoRA
checkpoint, generates held-out evaluation images from the prompt manifests,
and writes a generation_records.csv per run.

## Files to create
1. src/rd_lora/eval/__init__.py (empty)
2. src/rd_lora/eval/generate.py
3. scripts/generate_rdlora_eval_images.py
4. tests/test_rdlora_eval_generation.py

Do NOT edit any existing files.

## src/rd_lora/eval/generate.py

Two public functions:

### load_eval_pipeline

```python
from pathlib import Path
from typing import Any

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
```

### generate_eval_images_for_run

```python
import pandas as pd
from pathlib import Path

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
```

Implementation notes:
- Import StableDiffusionXLPipeline from diffusers
- Import torch for dtype and generator
- Use train_summary.json for backend/task (fields: "backend", "task")
- Use checkpoint_info.json for checkpoint path (field: "checkpoint_path")
- The checkpoint_path points to a directory containing adapter_model.safetensors
  or similar PEFT artifacts. Use pipe.load_lora_weights(checkpoint_path) to load.
- After generation, call pipe.unload_lora_weights() and del pipe to free VRAM

## scripts/generate_rdlora_eval_images.py

CLI wrapper:
--run_dir           (required) Path to a single training run directory
--manifest          (required) Path to the evaluation prompt manifest CSV
--output_dir        (required) Where to write generated images + generation_records.csv
--base_model_id     (default: stabilityai/stable-diffusion-xl-base-1.0)
--guidance_scale    (default: 7.5)
--num_inference_steps (default: 30)
--device            (default: cuda:0)
--torch_dtype       (default: float16)

Behavior:
- Parse args
- Call generate_eval_images_for_run
- Print output path
- Exit 0 on success, 1 on failure

## tests/test_rdlora_eval_generation.py

This is a UNIT TEST that does NOT require GPU or real SDXL model.
Use monkeypatching/mocking to test the logic without actual inference.

Test the following:
1. generation_records.csv has correct columns:
   run_dir, backend, task, prompt_id, prompt, seed, expected_text, image_path
2. Image file naming follows pattern: <backend>__<task>__<prompt_id>__seed<seed>.png
3. Number of output rows matches manifest row count
4. Function raises error if checkpoint_path does not exist
5. Function raises error if train_summary.json is missing

Mock strategy:
- Create a fake run_dir with minimal train_summary.json and checkpoint_info.json
- Mock StableDiffusionXLPipeline to return a small dummy PIL image
- Mock pipe.load_lora_weights as noop
- Use a small 3-row manifest CSV as input
- Verify outputs without actual GPU inference

## Constraints
- Do NOT edit any existing files
- Do NOT import from adapter_factory, execution, or timestep_routing
- Use only: diffusers, torch, pandas, PIL, pathlib, json
- generation_records.csv must use the exact column names specified
- File naming must be deterministic and identical across runs
- All generation parameters (guidance_scale, num_inference_steps, scheduler)
  must be frozen and identical across backends

## Validation
After completion:
    python3 -c "import ast; ast.parse(open('src/rd_lora/eval/generate.py').read()); print('OK')"
    python3 -c "import ast; ast.parse(open('scripts/generate_rdlora_eval_images.py').read()); print('OK')"
    python3 -m pytest tests/test_rdlora_eval_generation.py -x -q 2>&1 | tail -10
Expected: syntax OK, all tests pass (without GPU).
