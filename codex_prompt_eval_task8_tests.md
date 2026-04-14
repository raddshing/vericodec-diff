Branch: pivot/rd-lora-diff
Module: M05.3b — Task 8: gate contract tests

## Objective
Create two new test files that verify the M05.3b gate contract:
1. score must exist as a long-form metric_name row
2. gate goes INVALID if score or wall_time_sec is missing

Three test files already exist and must NOT be edited:
- tests/test_rdlora_eval_generation.py (Task 2)
- tests/test_rdlora_eval_metrics_subject.py (Task 4)
- tests/test_rdlora_eval_metrics_style.py (Task 4)

## Files to create
1. tests/test_rdlora_primary_metric_binding.py
2. tests/test_rdlora_gate_invalid_on_missing_score.py

Do NOT edit any existing files.

## tests/test_rdlora_primary_metric_binding.py

Test that the eval pipeline produces long-form rows with metric_name=="score".

This is a UNIT TEST. Do NOT require GPU or real SDXL model.
Mock all heavy computation (image generation, CLIP, DINO, OCR).

### Test 1: subject metric JSON contains "score"
- Create a fake metrics dict like compute_subject_metrics would return
- Verify "score" key exists and is a float in [0, 1]

### Test 2: style metric JSON contains "score"
- Create a fake metrics dict like compute_style_metrics would return
- Verify "score" key exists and is a float in [0, 1]

### Test 3: long-form CSV includes metric_name == "score"
- Create a minimal set of long-form rows as eval_rdlora.py would produce
- Structure: list of dicts with task, backend, metric_name, metric_value
- Verify at least one row has metric_name == "score" for each backend
- Verify at least one row has task == "__aggregate__" and metric_name == "score"

### Test 4: no wide-row "score" assumption
- Create a sample baseline_metrics.json in long-form format
- Verify that reading it does NOT require a wide "score" column
- Verify the _extract_metric helper from write_rdlora_gate_memo.py works
  on long-form rows

Implementation:
```python
import json
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.eval.score import subject_score, style_score, backend_score
```

For Test 4, import _extract_metric by importing the gate module:
```python
# Add scripts to path to import the gate module
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
# Then import the function from write_rdlora_gate_memo
```
Or alternatively, replicate the extraction logic in the test to avoid
fragile script imports. Prefer the replication approach.

## tests/test_rdlora_gate_invalid_on_missing_score.py

Test that the gate script produces INVALID when score or wall_time is missing.

This is a UNIT TEST. Do NOT require GPU. Use fixture files on disk.

### Test 1: gate INVALID when score missing from baseline_metrics.json
- Create a tmp_path with minimal evaluation artifacts:
  - evaluation_summary.json with metric_names that do NOT include "score"
  - baseline_metrics.json with rows that have metric_name="train_loss_last" but NOT "score"
- Create minimal probe, surrogate, allocation dirs with valid summary JSONs
- Run write_rdlora_gate_memo.py main() or invoke via subprocess
- Verify gate_memo.json has gate_status == "INVALID"
- Verify invalid_reasons contains something about "score" or "primary_metric"

### Test 2: gate INVALID when wall_time_sec missing for overhead ratio
- Create a tmp_path with evaluation artifacts that HAVE score rows
  but do NOT have wall_time_sec rows for uniform backend
- Create minimal probe, surrogate, allocation dirs
- Run the gate script
- Verify gate_memo.json has gate_status == "INVALID"
- Verify invalid_reasons mentions "wall_time_sec"

### Test 3: gate VALID_NO_GO when all metrics present but thresholds fail
- Create evaluation artifacts with score rows for all backends
  (use small values like 0.1) and wall_time_sec rows
- Create surrogate_summary.json with held_out_spearman_rho=0.0
- Run the gate script
- Verify gate_status == "VALID_NO_GO" (not INVALID)
- This confirms that missing metrics = INVALID, while present-but-low = VALID_NO_GO

### Implementation notes for fixture creation
Each test needs these minimal dirs/files:

probe_dir:
  - probe_summary.json: {"wall_time_seconds": 10.0}
  - cell_utility.json: {"rows": [{"cell_id": "c1", "utility_score": 0.5}]}

surrogate_dir:
  - surrogate_summary.json: {
      "held_out_spearman_rho": 0.0,
      "top_5_precision": 1.0,
      "wall_time_seconds": 5.0
    }

allocation_dir:
  - allocation_summary.json: {"wall_time_seconds": 2.0}

evaluation_dir:
  - evaluation_summary.json: {
      "schema_version": "1.0",
      "tasks": ["subject_personalization", "style_domain"],
      "backends": ["uniform", "layer_only", "timestep_only", "proposed"],
      "metric_names": [...],
      "evaluated_run_dirs": [],
      "primary_metric": "score"
    }
  - baseline_metrics.json: {"schema_version": "1.0", "rows": [...]}

For each test, create run dirs with minimal run_provenance.json:
{
    "run_mode": "real_gpu",
    "torch_cuda_is_available": true,
    "peak_vram_mib": 9500,
    "used_gpu": true
}

Use tmp_path fixture for all temp dirs.

To invoke the gate script, use subprocess:
```python
import subprocess
result = subprocess.run(
    [sys.executable, str(REPO_ROOT / "scripts" / "write_rdlora_gate_memo.py"),
     "--probe_dir", str(probe_dir),
     "--surrogate_dir", str(surrogate_dir),
     "--allocation_dir", str(allocation_dir),
     "--evaluation_dir", str(evaluation_dir),
     "--output_dir", str(output_dir)],
    capture_output=True, text=True
)
```

## Constraints
- Do NOT edit any existing files
- Tests must NOT require GPU
- Tests must NOT require real SDXL model
- Use tmp_path for all fixtures
- Use subprocess to invoke gate script (avoids import complications)

## Validation
After completion:
    python3 -c "import ast; ast.parse(open('tests/test_rdlora_primary_metric_binding.py').read()); print('OK')"
    python3 -c "import ast; ast.parse(open('tests/test_rdlora_gate_invalid_on_missing_score.py').read()); print('OK')"
    python3 -m pytest tests/test_rdlora_primary_metric_binding.py tests/test_rdlora_gate_invalid_on_missing_score.py -x -q 2>&1 | tail -15
    python3 -m pytest tests/test_rdlora_eval_generation.py tests/test_rdlora_eval_metrics_subject.py tests/test_rdlora_eval_metrics_style.py -x -q 2>&1 | tail -10
Expected: all tests pass, no regressions.
