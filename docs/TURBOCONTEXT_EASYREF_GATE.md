# TurboContext EasyRef Gate

This repo-local wrapper adds a deterministic EasyRef smoke runner and a VRAM profiling sweep for the TurboContext pre-gate.

## Entrypoints

- `scripts/run_easyref_baseline.py`: runs a fixed-ref-count EasyRef smoke suite and writes PNG outputs plus `easyref_run_summary.json`, `easyref_run_records.csv`, and `resolved_config.yaml` under `outputs/baselines/easyref/<suite>/ref_count_<n>/`.
- `scripts/profile_refmem_vram.py`: sweeps `ref_count` over `{1,2,4,8}` and writes `easyref_refmem_profile.json`, `easyref_refmem_profile.csv`, and `resolved_config.yaml` under `outputs/metrics/turbocontext_gate/`.

## Measurement semantics

- `total_peak_vram_gb`: maximum peak VRAM observed across the instrumented EasyRef components for a run.
- `ref_tokens_cond_bytes` / `ref_tokens_uncond_bytes`: bytes for the conditional and unconditional reference-token tensors returned by `EasyRef.get_image_embeds()`.
- `persistent_ref_interface_bytes`: `ref_tokens_cond_bytes + ref_tokens_uncond_bytes`.
- `persistent_ref_interface_gb`: `persistent_ref_interface_bytes / 1024^3`.
- `persistent_ref_interface_share`: `persistent_ref_interface_gb / total_peak_vram_gb`.
- `reference_encode_peak_delta_gb`: the larger transient delta across the conditional and unconditional `get_image_embeds()` spans.
- `reference_encode_peak_vram_gb`: the larger absolute peak across the conditional and unconditional `get_image_embeds()` spans.

The profiler also stores per-component peaks in JSON for `cond_reference_encode`, `uncond_reference_encode`, `prompt_conditioning`, and `diffusion_decode`.
Reference-memory accounting excludes `pooled_prompt_embeds` and does not treat the full concatenated `prompt_embeds` buffers as reference memory.

## Optional verification

- `profiling.verify_ip_attention: true` adds JSON-only verification records for the first observed IP-attention calls.
- Each verification record stores `num_tokens`, `ip_hidden_states_shape`, and `ip_kv_materialization`.
- Stock EasyRef computes `ip_key` and `ip_value` inside each attention call, so the recorded materialization mode is `transient_in_call` rather than a persistent cache.

## Runtime notes

- The default config in `configs/turbocontext_easyref_baseline.yaml` points at the pinned EasyRef checkout and uses a one-item tiny suite built from repo-local upstream assets.
- The real backend expects the EasyRef checkpoint, SDXL base model, and Qwen2-VL weights to be locally accessible or downloadable from the configured ids.
- `--backend-kind mock` is available for deterministic smoke/tests when the full EasyRef stack is not installed.
