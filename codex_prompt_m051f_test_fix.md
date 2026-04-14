Branch: pivot/rd-lora-diff
File to edit: tests/test_training_execution_path.py
Do NOT edit any other file.

## Context

In execution.py (already committed as M05.1f, commit 4b2c9a8):
- `apply_adapter_specs` now returns `list[str]` of created adapter names
- `collect_trainable_parameters` now accepts `adapter_names: Sequence[str] | None = None` and collects params by matching adapter name in `unet.named_parameters()` instead of filtering by `requires_grad`
- `assert_adapter_param_counts(unet, adapter_names)` was added — it calls `unet.named_parameters()` and `p.numel()` for each adapter
- `_assert_optimizer_has_grad(optimizer)` checks that at least one param in the optimizer has `.grad is not None` after backward
- `FakeOptimizer.zero_grad` is now called with `set_to_none=True`

The test `test_real_execution_path_loads_components_and_runs_backward_step_checkpoint` fails because the fake harness does not satisfy these new contracts.

## Required changes (5 items)

### 1. FakeParam (around line 75) — ALREADY DONE, do not touch
Already has `name: str = ""`, `self.grad = None`, and `def numel(self) -> int: return 1`.

### 2. FakeModel
- In `__init__`, change `FakeParam(requires_grad=False)` to `FakeParam(requires_grad=False, name=f"base_param_{i}")` so each param has a unique name.
- Add a `named_parameters` method right after `parameters`:
    def named_parameters(self):
        return [(p.name, p) for p in self._parameters]

### 3. FakeAccelerator
- In `prepare`, store a reference to the model (any item that has `named_parameters`):
    def prepare(self, *items):
        for item in items:
            if hasattr(item, "named_parameters"):
                self._model = item
        return items
- In `backward`, after `self.backward_calls.append(loss)`, set `.grad = True` on all `requires_grad` params of the stored model:
        if hasattr(self, "_model"):
            for param in self._model.parameters():
                if param.requires_grad:
                    param.grad = True

### 4. fake_apply_adapter_specs (around line 282 inside the test function)
Change from:
    def fake_apply_adapter_specs(**kwargs) -> None:
        kwargs["unet"].parameters()[1].requires_grad = True
To:
    def fake_apply_adapter_specs(**kwargs) -> list[str]:
        param = kwargs["unet"].parameters()[1]
        param.requires_grad = True
        param.name = "unet.to_q.uniform_bank.lora_A.weight"
        return ["uniform_bank"]

### 5. FakeOptimizer.zero_grad
Change from:
    def zero_grad(self) -> None:
To:
    def zero_grad(self, set_to_none: bool = False) -> None:

## Validation
After changes, run:
    python3 -m pytest tests/test_training_execution_path.py -x -q
Expected: all tests pass.

Also run:
    python3 -m pytest tests/test_noop_band_execution.py tests/test_training_backends_actual_contract.py -x -q
Expected: 5 + 2 = 7 passed (no regressions).
