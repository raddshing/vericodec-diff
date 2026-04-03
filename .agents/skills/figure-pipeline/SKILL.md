# Skill: figure-pipeline

Use this skill when preparing paper figures.

## Workflow
1. Read `configs/figures.yaml`.
2. Reuse existing metrics JSON/CSV files whenever possible.
3. Keep figure scripts pure: input metrics in, figure out.
4. Export both PDF and SVG.
5. Use stable filenames that match the paper.
6. If a figure requires qualitative samples, record the selected sample IDs in the config.

## Guardrails
- No recomputation of model outputs inside figure scripts.
- No hidden filtering of failed samples.
- If a baseline is missing, annotate that directly in the figure/table pipeline.
