Branch: pivot/rd-lora-diff
Module: M05.3b — LoRA export hook + eval loader fix

## Objective
Add Diffusers-format LoRA export inside save_checkpoint so eval can load
with pipe.load_lora_weights(). Follow the exact pattern from the official
train_dreambooth_lora_sdxl.py example.

## Files to edit
1. src/rd_lora/training/execution.py
2. src/rd_lora/eval/generate.py

Do NOT edit any other files.

## 1. execution.py — add save/load hooks

### Imports to add at top of file
```python
from peft.utils import get_peft_model_state_dict
from diffusers.loaders.lora_pipeline import StableDiffusionXLPipeline as LoraExportPipeline
from peft.utils import set_peft_model_state_dict
```

IMPORTANT: These imports may fail at module level because execution.py
lazy-loads diffusers. So wrap them in the hook functions or import inside
the function body where they are used.

### Where to add hooks
In the execute_training_run function, AFTER accelerator.prepare() call
and BEFORE the training loop, register two hooks.

Find the line (approximately):
```python
components["unet"], optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
```

AFTER the prepare block, add the hook registration.

### save_model_hook implementation
```python
def save_model_hook(models, weights, output_dir):
    if not getattr(accelerator, "is_main_process", True):
        return
    from peft.utils import get_peft_model_state_dict
    from diffusers.utils import convert_state_dict_to_diffusers

    for model in models:
        if hasattr(model, "peft_config"):
            state_dict = get_peft_model_state_dict(model)
            unet_lora_layers = convert_state_dict_to_diffusers(state_dict)
            # Use the class method directly to avoid instantiating a full pipeline
            from diffusers import StableDiffusionXLPipeline as _ExportPipeline
            _ExportPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers,
            )
        # pop weights so accelerator doesn't save the model again in default format
        if weights:
            weights.pop()
```

### load_model_hook implementation
```python
def load_model_hook(models, input_dir):
    from peft.utils import set_peft_model_state_dict
    from diffusers.loaders.lora_pipeline import StableDiffusionLoraLoaderMixin
    from diffusers.utils import convert_unet_state_dict_to_peft

    while len(models) > 0:
        model = models.pop()
        if hasattr(model, "peft_config"):
            lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)
            unet_state_dict = {
                k.replace("unet.", ""): v
                for k, v in lora_state_dict.items()
                if k.startswith("unet.")
            }
            unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
            set_peft_model_state_dict(model, unet_state_dict, adapter_name="default")
```

### Register hooks
```python
accelerator.register_save_state_pre_hook(save_model_hook)
accelerator.register_load_state_pre_hook(load_model_hook)
```

### Do NOT change save_checkpoint
Keep the existing save_checkpoint function as-is. The hook will be called
automatically when accelerator.save_state() runs.

## 2. generate.py — fix load_lora_weights call

In load_eval_pipeline, change:
```python
pipe.load_lora_weights(str(checkpoint_path))
```
To:
```python
pipe.load_lora_weights(str(checkpoint_path), weight_name="pytorch_lora_weights.safetensors")
```

This tells Diffusers to look for the specific LoRA export file inside the
checkpoint directory, not the raw model.safetensors.

## Import availability check
The following imports must work in the rdlora-env:
- peft.utils.get_peft_model_state_dict
- peft.utils.set_peft_model_state_dict
- diffusers.utils.convert_state_dict_to_diffusers
- diffusers.utils.convert_unet_state_dict_to_peft
- diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin

If any of these import paths don't exist in the installed versions,
use alternative paths. The official example uses:
  from diffusers.loaders.lora_conversion_utils import convert_state_dict_to_diffusers
Check what's available and use the correct path.

## Constraints
- Do NOT change save_checkpoint function
- Do NOT change any manifest, adapter_factory, or timestep_routing code
- Do NOT change noop semantics
- Hooks must be defined inside execute_training_run (they need closure over
  accelerator and components)
- Keep all existing tests passing

## Validation
After completion:
    python3 -c "import ast; ast.parse(open('src/rd_lora/training/execution.py').read()); print('execution OK')"
    python3 -c "import ast; ast.parse(open('src/rd_lora/eval/generate.py').read()); print('generate OK')"
    python3 -m pytest tests/test_training_execution_path.py tests/test_noop_band_execution.py tests/test_rdlora_eval_generation.py -x -q 2>&1 | tail -10
    grep -n "save_model_hook\|load_model_hook\|register_save_state\|register_load_state\|pytorch_lora_weights" src/rd_lora/training/execution.py
    grep -n "pytorch_lora_weights" src/rd_lora/eval/generate.py
Expected: syntax OK, tests pass, hooks registered, weight_name present.
