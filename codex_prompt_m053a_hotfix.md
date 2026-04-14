Branch: pivot/rd-lora-diff
Module: M05.3a — eval/gate contract hotfix

## Problem
The gate script (write_rdlora_gate_memo.py) has two contract bugs:
1. Reads primary_metric="score" from baseline rows, but "score" does not exist.
   All backend metrics default to 0.0, producing fake scientific failures.
2. Reads wall_time from baseline rows, but training outputs have no wall_time.
   Denominator is 0/None, producing Infinity for overhead ratio.

## Files to edit
1. src/rd_lora/training/execution.py — add wall_time_sec to train_summary.json
2. scripts/eval_rdlora.py — surface wall_time_sec in long-form baseline metrics
3. scripts/write_rdlora_gate_memo.py — fix primary metric + overhead ratio reading

Do NOT edit adapter_factory.py, timestep_routing.py, manifests, or test files.

## Current facts
- Training outputs: run_provenance.json, train_summary.json, metrics.json, checkpoint_info.json
- train_summary.json currently has: allocation_manifest, backend, checkpoint_path, status, task, train_steps
- No wall_time_sec exists anywhere in training outputs
- baseline_metrics.json rows have: backend, task, train_loss_last, train_loss_mean, etc. No "score", no "wall_time_seconds"
- evaluation_summary.json metric_names: active_band_optimizer_steps, global_step, learning_rate_last, skipped_noop_band_batches, skipped_noop_band_fraction, train_loss_last, train_loss_mean
- primary_metric in config defaults to "score"

## Required changes

### 1. execution.py — add wall_time_sec to train_summary.json
- Record start time (time.monotonic()) before the training loop
- Record end time after the training loop
- Add "wall_time_sec": round(elapsed, 3) to the train_summary payload
- This is the canonical source of truth for training wall time

### 2. eval_rdlora.py — surface wall_time_sec from train_summary
- When reading each run dir, also read train_summary.json
- Add wall_time_sec to the metrics row for that run
- It should appear in baseline_metrics.json rows and baseline_metrics.csv

### 3. write_rdlora_gate_memo.py — three fixes

#### 3a. primary_metric validation
- After loading evaluation_summary, check if primary_metric is in evaluation_summary["metric_names"]
- If NOT present: add to invalid_reasons: "primary_metric '{name}' not found in evaluation metric_names"
- Do NOT silently default to 0.0

#### 3b. primary_metric reading from baseline rows
- Keep existing logic but: when primary_metric is missing from a backend row,
  do NOT use .get(primary_metric, 0.0)
- Instead: if the key is absent, add to invalid_reasons and set metric to None
- Only compute proposed_relative_vs_uniform if all required metrics are non-None

#### 3c. overhead ratio fix
- Read wall_time_sec from baseline_rows (now available via eval fix)
- Change _wall_time_for_backend to look for "wall_time_sec" key (from train_summary)
- If still None or 0: add to invalid_reasons: "uniform wall_time_sec missing or zero"
- Set probe_allocation_overhead_ratio to None, not Infinity
- In _status_and_summary: if any metric is None, treat as INVALID

## Constraints
- Do NOT change noop semantics, adapter wiring, or routing logic
- Do NOT change manifest schemas
- Do NOT invent a "score" metric — that is a future task
- Fail-closed: missing data = INVALID, not silent 0.0
- Keep changes minimal

## Validation
After changes:
    python3 -c "import ast; ast.parse(open('src/rd_lora/training/execution.py').read()); print('OK')"
    python3 -c "import ast; ast.parse(open('scripts/eval_rdlora.py').read()); print('OK')"
    python3 -c "import ast; ast.parse(open('scripts/write_rdlora_gate_memo.py').read()); print('OK')"
    python3 -m pytest tests/test_training_execution_path.py tests/test_noop_band_execution.py tests/test_training_backends_actual_contract.py -x -q
Expected: syntax OK, all existing tests pass.
