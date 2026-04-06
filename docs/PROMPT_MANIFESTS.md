# Prompt Manifest Schema

VeriCodec-Diff uses three canonical prompt manifests under `data/manifests/`:

- `prompt_manifest_kill.csv`: 240 prompts for the week-1 to week-3 kill gate, balanced at 48 prompts per locked category.
- `prompt_manifest_main.csv`: 500 prompts for the broader evaluation pool, balanced at 100 prompts per locked category.
- `prompt_manifest_hard.csv`: 150 hard-detail prompts, balanced at 30 prompts per locked category.

All three manifests use the same seven-column schema, in this exact order:

1. `prompt_id`: deterministic identifier in the form `<split>_<category_token>_<index>`.
2. `category`: one of `text-heavy`, `thin-boundary`, `repeated-pattern`, `faces`, or `mixed scenes`.
3. `stress_type`: the specific failure mode targeted inside the category.
4. `prompt`: the positive text prompt.
5. `negative_prompt`: the negative prompt paired with the row.
6. `seed`: per-row deterministic seed.
7. `split`: one of `kill`, `main`, or `hard`.

Category design stays aligned with the locked project taxonomy:

- `text-heavy`: scenes where readable text is central; every prompt includes an explicit quoted string for OCR evaluation.
- `thin-boundary`: scenes dominated by hairline contours, transparent rims, filigree, wires, or similarly fragile boundaries.
- `repeated-pattern`: scenes dominated by dense repetition such as grids, tiles, fabric, shelving, or facades.
- `faces`: portraits or face-centric scenes that stress microfeatures such as eyelashes, hairlines, pores, freckles, and teeth edges.
- `mixed scenes`: layered scenes with many small objects, occlusions, materials, and depth interactions.

The `hard` manifest does not introduce a new category. It reuses the same five categories but pushes harder-detail variants through more severe `stress_type` choices such as microtext, subpixel outlines, dense repetition, facial microfeatures, and clutter-plus-reflection mixtures.

Repo-local validation lives in [src/vericodec_diff/prompt_manifests.py](/home/dima/src/vericodec-diff/src/vericodec_diff/prompt_manifests.py), and the runnable entrypoint is [scripts/validate_prompt_manifests.py](/home/dima/src/vericodec-diff/scripts/validate_prompt_manifests.py).

