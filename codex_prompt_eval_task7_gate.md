Branch: pivot/rd-lora-diff
Module: M05.3b — Task 7: gate binding to long-form eval contract

## Objective
Rewrite scripts/write_rdlora_gate_memo.py so it reads score and wall_time_sec
from the long-form baseline_metrics.json rows (metric_name/metric_value format),
not from wide backend dicts. Enforce fail-closed semantics per advisor rules.

## File to edit
scripts/write_rdlora_gate_memo.py — FULL REWRITE (keep filename, replace contents)

## Current eval output format (from Task 6)
baseline_metrics.json rows are LONG-FORM:
{
    "task": "subject_personalization",
    "backend": "uniform",
    "metric_name": "score",
    "metric_value": 0.734,
    "run_dir": "/path/to/run",
    "source_json": "/path/to/metrics.json"
}

Aggregate rows have task="__aggregate__":
{
    "task": "__aggregate__",
    "backend": "uniform",
    "metric_name": "score",
    "metric_value": 0.766,
    "run_dir": "",
    "source_json": ""
}

## Hard rules from advisor (ALL MUST BE FOLLOWED)
1. NEVER use .get(primary_metric, 0.0) on wide backend dicts
2. Read "score" from long-form rows where metric_name == "score"
3. If "score" missing for any required backend, set gate_status = INVALID
4. If wall_time_sec missing for overhead computation, set gate_status = INVALID
5. Do NOT collapse missing metrics into VALID_NO_GO — missing = INVALID

## CLI arguments (keep same as current)
--config, --probe_dir, --surrogate_dir, --allocation_dir, --evaluation_dir, --output_dir

## New internal logic

### Reading long-form baseline metrics

```python
def _load_long_form_rows(evaluation_dir: Path) -> list[dict[str, Any]]:
    payload = _read_json(evaluation_dir / "baseline_metrics.json")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("baseline_metrics.json must contain rows")
    return [dict(row) for row in rows]

def _extract_metric(
    rows: list[dict[str, Any]],
    *,
    backend: str,
    task: str,
    metric_name: str,
) -> float | None:
    """Find a specific metric from long-form rows. Return None if not found."""
    for row in rows:
        if (str(row.get("backend")) == backend
            and str(row.get("task")) == task
            and str(row.get("metric_name")) == metric_name):
            val = row.get("metric_value")
            if val is not None and val != "":
                return float(val)
    return None
```

### Backend score extraction
For each required backend, read __aggregate__ score:
```python
score = _extract_metric(rows, backend=backend, task="__aggregate__", metric_name="score")
```
If None for any required backend -> invalid_reasons.append(...)

### proposed_relative_vs_uniform computation
```python
score_proposed = _extract_metric(rows, backend="proposed", task="__aggregate__", metric_name="score")
score_uniform = _extract_metric(rows, backend="uniform", task="__aggregate__", metric_name="score")
if score_proposed is not None and score_uniform is not None:
    proposed_relative_vs_uniform = round(
        (score_proposed - score_uniform) / max(1e-8, abs(score_uniform)), 6
    )
else:
    proposed_relative_vs_uniform = None
```

### proposed >= best(layer_only, timestep_only)
```python
score_layer_only = _extract_metric(rows, backend="layer_only", task="__aggregate__", metric_name="score")
score_timestep_only = _extract_metric(rows, backend="timestep_only", task="__aggregate__", metric_name="score")
best_ablation = max(score_layer_only, score_timestep_only) if both not None else None
proposed_ge_best = score_proposed >= best_ablation if both not None else None
```

### probe_allocation_overhead_ratio computation
Advisor formula:
```python
# Numerator: probe + surrogate + allocation wall times
overhead_numerator = (
    float(probe_summary.get("wall_time_seconds", 0.0))
    + float(surrogate_summary.get("wall_time_seconds", 0.0))
    + float(allocation_summary.get("wall_time_seconds", 0.0))
)

# Denominator: sum of uniform wall_time_sec across BOTH tasks
uniform_wall_task1 = _extract_metric(
    rows, backend="uniform", task="subject_personalization", metric_name="wall_time_sec"
)
uniform_wall_task2 = _extract_metric(
    rows, backend="uniform", task="style_domain", metric_name="wall_time_sec"
)

if uniform_wall_task1 is not None and uniform_wall_task2 is not None:
    denominator = uniform_wall_task1 + uniform_wall_task2
    if denominator <= 0.0:
        invalid_reasons.append("uniform wall_time_sec sum is zero")
        probe_allocation_overhead_ratio = None
    else:
        probe_allocation_overhead_ratio = round(overhead_numerator / denominator, 6)
else:
    invalid_reasons.append("uniform wall_time_sec missing for one or both tasks")
    probe_allocation_overhead_ratio = None
```

### _status_and_summary (keep same logic)
If invalid_reasons or any metric value is None -> INVALID
Otherwise check thresholds -> VALID_GO or VALID_NO_GO

### Output format (keep same)
gate_memo.json with: gate_status, thresholds, metrics, source_run_dirs, invalid_reasons, decision_summary
gate_memo.md with human-readable summary

## Things to keep from current script
- parse_args (same CLI)
- _default_config (same thresholds and policies)
- _load_config
- _read_json
- _top_20_mass_ratio (reads from probe cell_utility.json, unchanged)
- provenance validation loop (unchanged)
- resolved_config writing (unchanged)
- gate_memo.md format (unchanged)

## Things to REMOVE
- _primary_metric_rows (replaced by _load_long_form_rows)
- _metric_for_backend (replaced by _extract_metric using long-form)
- _wall_time_for_backend (replaced by _extract_metric)
- Any .get(primary_metric, 0.0) pattern
- Any wide-row baseline_by_backend dict usage for score/wall_time

## Constraints
- FULL REWRITE of write_rdlora_gate_memo.py
- Keep same filename
- Do NOT edit any other files
- Do NOT use .get(..., 0.0) for gate-critical metrics
- Missing metrics = INVALID, never VALID_NO_GO
- Long-form only: metric_name/metric_value rows

## Validation
After completion:
    python3 -c "import ast; ast.parse(open('scripts/write_rdlora_gate_memo.py').read()); print('OK')"
    python3 scripts/write_rdlora_gate_memo.py --help
    grep -c "get(primary_metric, 0.0)" scripts/write_rdlora_gate_memo.py
    grep -n "_extract_metric\|__aggregate__\|long.form\|metric_name\|metric_value" scripts/write_rdlora_gate_memo.py | head -20
    python3 -m pytest tests/test_training_execution_path.py tests/test_noop_band_execution.py tests/test_rdlora_eval_manifests.py tests/test_rdlora_eval_generation.py tests/test_rdlora_eval_metrics_subject.py tests/test_rdlora_eval_metrics_style.py -x -q 2>&1 | tail -10
Expected: syntax OK, help prints, zero occurrences of .get(primary_metric, 0.0), _extract_metric present, all existing tests pass.
