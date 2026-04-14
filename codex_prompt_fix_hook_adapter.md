Branch: pivot/rd-lora-diff
File to edit: src/rd_lora/training/execution.py
Do NOT edit any other file.

## Problem
save_model_hook calls get_peft_model_state_dict(model) without specifying
adapter_name. PEFT defaults to adapter_name="default", but our project uses
custom adapter names like "uniform_bank", "timestep_only__timestep_band_02", etc.
This causes KeyError: 'default' on every save_state() call.

## Fix
In the save_model_hook function (around line 1248), change the single
get_peft_model_state_dict(model) call to iterate over all adapter names
in model.peft_config and merge their state dicts.

Change this block:
    if hasattr(model, "peft_config"):
        state_dict = get_peft_model_state_dict(model)
        unet_lora_layers = convert_state_dict_to_diffusers(state_dict)

To:
    if hasattr(model, "peft_config") and model.peft_config:
        merged_state_dict = {}
        for adapter_name in model.peft_config:
            adapter_sd = get_peft_model_state_dict(model, adapter_name=adapter_name)
            merged_state_dict.update(adapter_sd)
        unet_lora_layers = convert_state_dict_to_diffusers(merged_state_dict)

Keep everything else in the function exactly as-is (imports, save_lora_weights
call, weights.pop(), etc).

## Constraints
- Only change the 3 lines described above
- Do NOT change save_checkpoint, load_model_hook, or anything else
- Do NOT edit any other file

## Validation
    python3 -c "import ast; ast.parse(open('src/rd_lora/training/execution.py').read()); print('OK')"
    grep -A5 "peft_config" src/rd_lora/training/execution.py | grep -c "adapter_name"
    python3 -m pytest tests/test_training_execution_path.py tests/test_noop_band_execution.py -x -q
Expected: syntax OK, adapter_name appears, tests pass.
