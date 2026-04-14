Branch: pivot/rd-lora-diff
Module: M05.1g — deterministic smoke routing harness

## Objective
Add a smoke-only forced routing mechanism so the M05.1f GPU smoke ladder can
deterministically test specific timestep-band transitions. This is test/debug
infrastructure only — do NOT touch allocator logic, manifests, or zero-rank
noop semantics.

## Files to edit
1. src/rd_lora/training/timestep_routing.py — add a function to resolve a
   forced band name to a deterministic in-band timestep value
2. src/rd_lora/training/execution.py — consume a forced band sequence and
   stop when exhausted
3. tests/test_forced_smoke_routing.py — new test file

Do NOT edit adapter_factory.py, allocation_manifest.py, or any manifest JSON.

## Current architecture (read these files first)
- src/rd_lora/training/timestep_routing.py (full file)
- src/rd_lora/training/execution.py (focus on lines 674-740 and 1158-1260)
- scripts/train_rdlora.py (the CLI entry point)

## Key facts

### Cell schema (20 reference steps, num_train_timesteps=1000)
- timestep_band_00: step_indices [0..4], timestep_values [15..19]
- timestep_band_01: step_indices [5..9], timestep_values [10..14]
- timestep_band_02: step_indices [10..14], timestep_values [5..9]
- timestep_band_03: step_indices [15..19], timestep_values [0..4]

### Timestep projection (project_training_timestep_to_step_index)
- progress = (999 - timestep) / 999
- step_index = round(progress * 19)
- So training timestep ~900 maps to step_index ~2 (band_00)
- Training timestep ~350 maps to step_index ~12 (band_02)
- Training timestep ~50 maps to step_index ~18 (band_03)

### Manifest band layout (both timestep_only and proposed)
- band_00: rank=0 (noop)
- band_01: rank=0 (noop)
- band_02: rank>0 (trainable)
- band_03: rank>0 (trainable)

## Required changes

### 1. timestep_routing.py — add resolve_deterministic_timestep_for_band()

Add a function that given a band name and the routing table, returns one
deterministic training timestep (int in [0,999]) that is guaranteed to route
to that band. Use the middle step_index of the band and reverse the
projection: pick the timestep that maps to that step_index.

```python
def resolve_deterministic_timestep_for_band(
    routing_table: Mapping[str, Mapping[str, Sequence[int]]],
    band_name: str,
    *,
    num_train_timesteps: int = 1000,
    reference_step_count: int = 20,
) -> int:
```

Add to __all__.

### 2. execution.py — forced_timestep_band_sequence support

#### 2a. In _plan_timestep_band_batch_route (around line 701)
Add an optional parameter `forced_timestep: int | None = None`. When provided,
use it instead of calling _sample_timestep_band_training_timesteps. Create a
tensor of shape (batch_size,) filled with that value.

#### 2b. In the training loop (around line 1158)
Accept `forced_timestep_band_sequence` from the plan or from a new kwarg to
execute_training_run. When present:
- Before each batch, pop the next band name from the sequence
- Call resolve_deterministic_timestep_for_band() to get the forced timestep
- Pass it to _plan_timestep_band_batch_route as forced_timestep
- When the sequence is exhausted, break the training loop (stop condition)

#### 2c. In execute_training_run signature
Add optional parameter:
  forced_timestep_band_sequence: Sequence[str] | None = None
Pass it through. When provided, it overrides max_train_steps as the stop
condition (loop ends when sequence is exhausted).

#### 2d. In parse_args (scripts/train_rdlora.py)
Add --forced_timestep_band_sequence as an optional comma-separated argument.
Parse to list[str] or None.

### 3. tests/test_forced_smoke_routing.py

Write tests that verify:
- resolve_deterministic_timestep_for_band returns a timestep that actually
  routes back to the requested band (round-trip test for each of the 4 bands)
- A forced sequence of [band_00, band_02] with the timestep_only routing
  table produces: batch 1 routed to band_00 (noop), batch 2 routed to
  band_02 (trainable)

## Constraints
- Do NOT change noop semantics (zero-rank bands skip backward/optimizer/scheduler)
- Do NOT change adapter_factory.py
- Do NOT change any manifest JSON
- Do NOT change how non-forced (normal stochastic) routing works
- forced routing is purely additive — when forced_timestep_band_sequence is
  None, everything works exactly as before
- Keep the patch minimal

## Validation
After changes, run:
    PYTHONPATH=src python3 -m pytest tests/test_forced_smoke_routing.py -x -q
    python3 -m pytest tests/test_noop_band_execution.py tests/test_training_backends_actual_contract.py tests/test_training_execution_path.py -x -q
Expected: all tests pass, no regressions.
Also:
    python3 -c "import ast; ast.parse(open('src/rd_lora/training/execution.py').read()); print('OK')"
    python3 -c "import ast; ast.parse(open('src/rd_lora/training/timestep_routing.py').read()); print('OK')"
