Branch: pivot/rd-lora-diff
File to edit: src/rd_lora/training/execution.py
Do NOT edit any other file.

## Context
GPU smoke failed with: ValueError: Attempting to unscale FP16 gradients.
The adapter params entering the optimizer are fp16 because cast_training_params
uses requires_grad filtering, not adapter-name filtering.

## Required changes (3 items)

### 1. Add two helper functions after assert_adapter_param_counts

Add upcast_trainable_params_to_fp32(torch_module, trainable_parameters):
- Iterate params, skip duplicates by id()
- If param.is_floating_point() and param.dtype != torch_module.float32,
  cast param.data to float32
- If param.grad exists and is floating point, cast it too

Add assert_all_params_fp32(torch_module, trainable_parameters):
- Raise TrainingExecutionError if any floating-point param is not float32

Add both to __all__.

### 2. Change the callsite (around line 1173)

Current order:
    if training_args.mixed_precision == "fp16":
        official_module.cast_training_params([components["unet"]], dtype=torch_module.float32)
    trainable_parameters = collect_trainable_parameters(components["unet"], adapter_names=created_adapter_names)
    optimizer = create_optimizer(

New order:
    trainable_parameters = collect_trainable_parameters(components["unet"], adapter_names=created_adapter_names)
    if training_args.mixed_precision == "fp16":
        upcast_trainable_params_to_fp32(torch_module, trainable_parameters)
    assert_all_params_fp32(torch_module, trainable_parameters)
    optimizer = create_optimizer(

Remove the cast_training_params line entirely. Replace with the new sequence.

## Constraints
- Do NOT edit any other file
- Do NOT change adapter_factory.py, timestep_routing.py, or manifests
- Do NOT change noop semantics or routing logic

## Validation
    python3 -c "import ast; ast.parse(open('src/rd_lora/training/execution.py').read()); print('OK')"
    python3 -m pytest tests/test_training_execution_path.py tests/test_noop_band_execution.py tests/test_training_backends_actual_contract.py -x -q
Expected: syntax OK, all tests pass.
