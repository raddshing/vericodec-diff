# TurboContext EasyRef Gate

This repo-local wrapper adds a deterministic EasyRef smoke runner and a VRAM profiling sweep for the TurboContext pre-gate.

## Entrypoints

- `scripts/run_easyref_baseline.py`: runs a fixed-ref-count EasyRef smoke suite and writes PNG outputs plus `easyref_run_summary.json`, `easyref_run_records.csv`, and `resolved_config.yaml` under `outputs/baselines/easyref/<suite>/ref_count_<n>/`.
- `scripts/profile_refmem_vram.py`: sweeps `ref_count` over `{1,2,4,8}` and writes `easyref_refmem_profile.json`, `easyref_refmem_profile.csv`, and `resolved_config.yaml` under `outputs/metrics/turbocontext_gate/`.

## Measurement semantics

- `total_peak_vram_gb`: maximum peak VRAM observed across the instrumented EasyRef components for a run.
- `refmem_peak_vram_gb`: persistent conditioning memory measured from the resident CUDA tensor bytes held after prompt/reference preparation and before diffusion.
- `refmem_share`: `refmem_peak_vram_gb / total_peak_vram_gb`.

The profiler also stores per-component peaks in JSON for `cond_reference_encode`, `uncond_reference_encode`, `prompt_conditioning`, and `diffusion_decode`.

## Runtime notes

- The default config in `configs/turbocontext_easyref_baseline.yaml` points at the pinned EasyRef checkout and uses a one-item tiny suite built from repo-local upstream assets.
- The real backend expects the EasyRef checkpoint, SDXL base model, and Qwen2-VL weights to be locally accessible or downloadable from the configured ids.
- `--backend-kind mock` is available for deterministic smoke/tests when the full EasyRef stack is not installed.
