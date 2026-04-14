Branch: pivot/rd-lora-diff
Module: M05.3b — Task 5: metric orchestration script

## Objective
Create a CLI orchestration script that dispatches to the correct task-specific
metric calculator and writes a per-run JSON metrics file. This is the bridge
between image generation (Task 2) and eval aggregation (Task 6).

## Files to create
1. scripts/compute_rdlora_eval_metrics.py

Do NOT edit any existing files.

## scripts/compute_rdlora_eval_metrics.py

### CLI arguments
--config              YAML config path (default: configs/rdlora_gate.yaml)
--run_dir             Path to a single training run directory (required)
--task                One of: subject_personalization, style_domain (required)
--generated_records_csv  Path to generation_records.csv from Task 2 (required)
--output_json         Path to write the per-run metrics JSON (required)
--reference_image_dir Path to reference images (required for subject_personalization,
ignored for style_domain)
--device              GPU device (default: cuda:0)
--ocr_engine          OCR engine for style_domain (default: tesseract)
### Behavior

1. Parse CLI arguments
2. Validate that generated_records_csv exists; fail closed if not
3. Validate that run_dir exists and contains train_summary.json; fail closed if not
4. Read train_summary.json to get "backend" and "task"
5. Verify that --task matches train_summary["task"]; fail closed if mismatch
6. Dispatch based on --task:
   - If "subject_personalization":
     a. Validate --reference_image_dir is provided and exists; fail closed if not
     b. Call compute_subject_metrics(
            generated_records_csv=Path(args.generated_records_csv),
            reference_image_dir=Path(args.reference_image_dir),
            device=args.device,
        )
   - If "style_domain":
     a. Call compute_style_metrics(
            generated_records_csv=Path(args.generated_records_csv),
            device=args.device,
            ocr_engine=args.ocr_engine,
        )
7. Build output JSON payload:
   {
       "run_dir": str(absolute path of run_dir),
       "backend": str(from train_summary),
       "task": str(from --task),
       "num_images": int(from metrics result),
       ...all task-specific metric fields from the result dict...,
       "score": float(from metrics result),
   }
8. Write output_json with json.dump, indent=2
9. Print: f"metrics_json={output_json}"
10. Print: f"task={task}"
11. Print: f"backend={backend}"
12. Print: f"score={score}"
13. Exit 0 on success, 1 on failure

### Per-run JSON schema

For subject_personalization:
{
    "run_dir": "/absolute/path/to/run_dir",
    "backend": "uniform",
    "task": "subject_personalization",
    "num_images": 32,
    "dino_ref_max_mean": 0.75,
    "clip_i_ref_max_mean": 0.82,
    "clip_t_mean": 0.88,
    "score": 0.734
}

For style_domain:
{
    "run_dir": "/absolute/path/to/run_dir",
    "backend": "uniform",
    "task": "style_domain",
    "num_images": 32,
    "ocr_exact_match": 0.6,
    "ocr_cer_mean": 0.15,
    "ocr_text_score": 0.775,
    "clip_t_mean": 0.85,
    "score": 0.7975
}

### Imports

```python
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rd_lora.eval.metrics_subject import compute_subject_metrics
from rd_lora.eval.metrics_style import compute_style_metrics
```

### Error handling

- All validation failures should print a clear error message to stderr and sys.exit(1)
- Do not catch and silence exceptions from metric computation
- If train_summary.json is malformed, print error and exit 1

## Tests

Do NOT create a separate test file for Task 5.
The orchestration script will be tested end-to-end in Task 8
(test_rdlora_primary_metric_binding.py).

However, the script must be syntactically valid and importable.

## Constraints
- Do NOT edit any existing files
- Do NOT import from adapter_factory, execution, or timestep_routing
- Script must be standalone — no dependency on eval_rdlora.py
- Fail closed on any missing input
- JSON output must be deterministic (sorted keys)

## Validation
After completion:
    python3 -c "import ast; ast.parse(open('scripts/compute_rdlora_eval_metrics.py').read()); print('OK')"
    python3 scripts/compute_rdlora_eval_metrics.py --help
    python3 -m pytest tests/test_training_execution_path.py tests/test_noop_band_execution.py tests/test_rdlora_eval_manifests.py tests/test_rdlora_eval_generation.py tests/test_rdlora_eval_metrics_subject.py tests/test_rdlora_eval_metrics_style.py -x -q 2>&1 | tail -10
Expected: syntax OK, help prints without error, all existing tests pass.
