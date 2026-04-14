Branch: pivot/rd-lora-diff
Module: M05.3b — Task 6: integrate generation + metrics into eval_rdlora.py

## Objective
Rewrite scripts/eval_rdlora.py to be the single M05.3 evaluation entrypoint.
For every valid real-GPU run: verify provenance, locate checkpoint, generate
held-out images (if cache missing), compute per-run metrics, and aggregate
into long-form baseline_metrics.csv.

## File to edit
scripts/eval_rdlora.py — FULL REWRITE (keep the file, replace contents)

## New CLI arguments (add to existing)
--config                  (default: configs/rdlora_gate.yaml)
--run_dir                 (repeatable, one or more run dirs)
--output_dir              (required)
--subject_manifest        (default: data/rd_lora/eval/subject_personalization_eval.csv)
--style_manifest          (default: data/rd_lora/eval/style_domain_eval.csv)
--reference_image_dir     (default: tests/fixtures/rdlora_pilot)
--base_model_id           (default: stabilityai/stable-diffusion-xl-base-1.0)
--guidance_scale          (default: 7.5)
--num_inference_steps     (default: 30)
--device                  (default: cuda:0)
--ocr_engine              (default: tesseract)
--skip_generation         (flag: if set, assume images already exist in cache)

## New behavior (7 steps per run_dir)

For each run_dir:

### Step 1: verify real-GPU provenance
- Load run_provenance.json
- Call assert_real_gpu_provenance
- Check no forbidden tokens in paths
- Fail closed on any issue

### Step 2: locate checkpoint
- Read checkpoint_info.json to get checkpoint_path
- If checkpoint_info.json missing, fall back to train_summary.json checkpoint_path
- Verify checkpoint_path exists on disk
- Fail closed if not

### Step 3: read task and backend
- Read train_summary.json
- Extract backend and task

### Step 4: generate held-out images (if cache missing)
- Cache dir: output_dir / "generated" / f"{backend}__{task}"
- generation_records.csv path: cache_dir / "generation_records.csv"
- If generation_records.csv already exists AND --skip_generation not set,
  regenerate anyway (overwrite)
- If --skip_generation is set and generation_records.csv exists, skip
- Select manifest based on task:
  subject_personalization -> args.subject_manifest
  style_domain -> args.style_manifest
- Call generate_eval_images_for_run from rd_lora.eval.generate
- Pass: run_dir, manifest path, cache_dir, base_model_id, guidance_scale,
  num_inference_steps, device, torch_dtype="float16"

### Step 5: compute per-run metrics
- metrics_json_path: cache_dir / "metrics.json"
- Dispatch based on task:
  subject_personalization -> compute_subject_metrics(
      generated_records_csv, reference_image_dir, device)
  style_domain -> compute_style_metrics(
      generated_records_csv, device, ocr_engine)
- Write metrics dict to metrics_json_path with json.dump

### Step 6: read wall_time_sec
- From train_summary.json field "wall_time_sec"
- If missing or not a number, set to None (do not use filesystem timestamps)

### Step 7: collect into long-form rows
- For each metric in the per-run metrics dict (except "num_images"):
  append a row: {task, backend, metric_name, metric_value, run_dir, source_json}
- Also append wall_time_sec row if available:
  {task, backend, metric_name="wall_time_sec", metric_value, run_dir, source_json}

After all runs processed:

### Aggregate rows
- For each backend, compute backend_score from the two task scores
  using rd_lora.eval.score.backend_score
- Append aggregate rows with task="__aggregate__", metric_name="score"
- For wall_time_sec aggregation: sum wall_time_sec across tasks for each backend
  Append with task="__aggregate__", metric_name="wall_time_sec"
  If any task is missing wall_time_sec, set aggregate to None and skip row

### Write outputs

baseline_metrics.csv — LONG FORM:
Columns: task, backend, metric_name, metric_value, run_dir, source_json
Sorted by: (backend, task, metric_name)

baseline_metrics.json:
{
    "schema_version": "1.0",
    "rows": [same rows as CSV but as list of dicts]
}

evaluation_summary.json:
{
    "schema_version": "1.0",
    "tasks": sorted list of tasks,
    "backends": sorted list of backends,
    "metric_names": sorted list of unique metric_names,
    "evaluated_run_dirs": list of run dirs,
    "primary_metric": "score",
    "wall_time_seconds": elapsed time of this eval script
}

## Imports needed

```python
from rd_lora.eval.generate import generate_eval_images_for_run
from rd_lora.eval.metrics_subject import compute_subject_metrics
from rd_lora.eval.metrics_style import compute_style_metrics
from rd_lora.eval.score import backend_score
```

## Key rules
- baseline_metrics.csv must be LONG FORM (task, backend, metric_name, metric_value)
- NOT wide form (one column per metric)
- "score" must appear as metric_name="score" rows
- wall_time_sec source: train_summary.json only, never filesystem timestamps
- if wall_time_sec missing, skip that row but do not fail the eval
- __aggregate__ rows must be present for gate script consumption
- fail closed on: missing provenance, missing checkpoint, forbidden tokens
- do not fail on: missing wall_time_sec (just skip those rows)

## Constraints
- This is a FULL REWRITE of eval_rdlora.py
- Keep the same filename: scripts/eval_rdlora.py
- Do NOT edit any other files
- Do NOT import from adapter_factory, execution, or timestep_routing
- Do NOT use filesystem timestamps for wall_time
- All image generation and metric computation happens inside this script

## Validation
After completion:
    python3 -c "import ast; ast.parse(open('scripts/eval_rdlora.py').read()); print('OK')"
    python3 scripts/eval_rdlora.py --help
    python3 -m pytest tests/test_training_execution_path.py tests/test_noop_band_execution.py tests/test_rdlora_eval_manifests.py tests/test_rdlora_eval_generation.py tests/test_rdlora_eval_metrics_subject.py tests/test_rdlora_eval_metrics_style.py -x -q 2>&1 | tail -10
Expected: syntax OK, help prints, all existing tests pass.
