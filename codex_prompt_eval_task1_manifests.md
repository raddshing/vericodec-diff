Branch: pivot/rd-lora-diff
Module: M05.3b — Task 1: held-out evaluation prompt manifests

## Objective
Create two deterministic CSV manifests for held-out evaluation image generation.
These manifests define exactly which prompts and seeds are used to evaluate every
backend checkpoint. They must be identical across all backends.

## Files to create
1. data/rd_lora/eval/subject_personalization_eval.csv
2. data/rd_lora/eval/style_domain_eval.csv
3. scripts/validate_rdlora_eval_manifests.py
4. tests/test_rdlora_eval_manifests.py

Do NOT edit any existing files.

## Manifest 1: subject_personalization_eval.csv

Columns: prompt_id, task, prompt, seed, expected_text

Rules:
- task = "subject_personalization" for all rows
- 8 held-out prompts about "the subject" (a dog) in varied settings
- 4 fixed seeds per prompt: 42, 123, 456, 789
- total: 32 rows
- expected_text is empty string for all rows
- prompt_id format: "subj_01" through "subj_08"
- prompts should be realistic DreamBooth-style prompts, e.g.:
  "A photo of the subject sitting on a park bench"
  "A studio portrait of the subject with dramatic lighting"
  "The subject wearing a red bandana in a garden"
  etc.
- Use "sks" as the rare token identifier in prompts, e.g.:
  "A photo of sks dog sitting on a park bench"
- Deterministic row order: sorted by (prompt_id, seed)

## Manifest 2: style_domain_eval.csv

Columns: prompt_id, task, prompt, seed, expected_text

Rules:
- task = "style_domain" for all rows
- 16 held-out signage prompts with explicit quoted text
- 2 fixed seeds per prompt: 42, 123
- total: 32 rows
- expected_text MUST be non-empty for every row
- expected_text is the exact text that should appear on the sign
- prompt_id format: "sign_01" through "sign_16"
- prompts should describe synthetic signs with quoted text, e.g.:
  "A wooden shop sign that reads 'OPEN DAILY'"
  "A neon restaurant sign displaying 'PIZZA HUT'"
  "A metal street sign saying 'NO PARKING'"
  etc.
- expected_text for the above examples: "OPEN DAILY", "PIZZA HUT", "NO PARKING"
- Deterministic row order: sorted by (prompt_id, seed)

## Validator: scripts/validate_rdlora_eval_manifests.py

Create a script that validates both manifests:
- Reads CSV with pandas
- Checks required columns exist
- Checks no duplicate (prompt_id, seed) pairs
- Checks row counts: 32 each
- Checks expected_text is empty for subject task, non-empty for style task
- Checks all seeds are in {42, 123, 456, 789} for subject, {42, 123} for style
- Checks deterministic sort order
- Exits 0 on success, 1 on failure with descriptive error

CLI:
    python3 scripts/validate_rdlora_eval_manifests.py \
      --subject data/rd_lora/eval/subject_personalization_eval.csv \
      --style data/rd_lora/eval/style_domain_eval.csv

## Test: tests/test_rdlora_eval_manifests.py

Test that:
- Both CSV files exist and parse without error
- Required columns present
- No duplicate (prompt_id, seed)
- Row counts correct (32 each)
- expected_text rules enforced
- Seed sets correct
- Sort order is deterministic

## Constraints
- Do NOT edit any existing files
- Do NOT create any Python packages (no __init__.py in data/)
- Use pandas for CSV reading in validator/tests
- Manifests are static data files, not generated

## Validation
After completion:
    python3 scripts/validate_rdlora_eval_manifests.py \
      --subject data/rd_lora/eval/subject_personalization_eval.csv \
      --style data/rd_lora/eval/style_domain_eval.csv
    python3 -m pytest tests/test_rdlora_eval_manifests.py -x -q
Expected: validator exits 0, all tests pass.
