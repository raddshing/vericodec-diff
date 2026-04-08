# RD-LoRA baseline status

Audit date: 2026-04-09

## Status definitions

- `available`: official code checkout is present locally and upstream documents a concrete run path.
- `appendix-only`: official code checkout is present locally, but the official runnable path stays outside the SDXL critical path.
- `partial`: official checkout exists, but the documented official run path still requires manual correction or patching before a clean run.
- `not-used`: an official or literature baseline was reviewed and intentionally left out of the runnable week-2 path.
- `missing`: no official runnable repository was verified from an official source on 2026-04-08.
- `literature_only`: a paper or official project page exists, but no official runnable repository was verified on 2026-04-08.

## Baseline summary

| name | official_repo | local_path | status | critical_path | notes |
| --- | --- | --- | --- | --- | --- |
| diffusers SDXL LoRA | `https://github.com/huggingface/diffusers.git` | `baselines/external/huggingface_diffusers` | `available` | `yes` | Official SDXL LoRA examples are present locally. The pinned main checkout requires `diffusers>=0.38.0.dev0`; the text-to-image example also moved to newer `peft` than the T-LoRA base environment. |
| T-LoRA | `https://github.com/ControlGenAI/T-LoRA.git` | `baselines/external/controlgenai_t-lora` | `available` | `yes` | Official SDXL path uses `stabilityai/stable-diffusion-xl-base-1.0`, `--resolution=1024`, and the shipped `tlora_env.yml`. This is the preferred RD-LoRA critical-path personalization baseline and the only required external week-2 baseline. |
| IntLoRA | `https://github.com/csguoh/IntLoRA.git` | `baselines/external/csguoh_intlora` | `appendix-only` | `no` | Official repo and environment are verified, but the official runnable path is quantized DreamBooth on Stable Diffusion 1.5. Do not block the SDXL week-2 gate on IntLoRA unless an official SDXL path is obviously trivial. |
| TSM | `not_verified` | `n/a` | `literature_only` | `no` | Interpreted here as TimeStep Master. No official runnable repository was verified on 2026-04-08. |
| AIRA | `not_verified` | `n/a` | `missing` | `no` | No official runnable repository could be verified from an official source on 2026-04-08, so AIRA stays out of the runnable registry. |
| MPQ-DM | `https://github.com/wlfeng0509/MPQ-DM.git` | `baselines/external/wlfeng0509_mpq-dm` | `partial` | `no` | Official repo exists locally, but the README still requires a manual `ddpm.py` modification and targets LDM `cin256-v2`, not SDXL. |
| AccuQuant | `not_verified` | `n/a` | `literature_only` | `no` | The official project page is public, but no official code repository was verified on 2026-04-08. |

## Verified registered baselines

| Baseline | Branch | Pinned commit | Local checkout | Expected model / checkpoint dependencies | Entrypoint notes |
| --- | --- | --- | --- | --- | --- |
| diffusers SDXL LoRA | `main` | `431066e96762442aad5b675893a91bb8c5bfb3b9` | `baselines/external/huggingface_diffusers` | `stabilityai/stable-diffusion-xl-base-1.0` | Official examples are `examples/text_to_image/train_text_to_image_lora_sdxl.py` and `examples/dreambooth/train_dreambooth_lora_sdxl.py`. Both save SDXL LoRA weights through `StableDiffusionXLPipeline.save_lora_weights(...)`. |
| T-LoRA | `main` | `12f36f361bbb6e32e3bb3efc95bdf60db9b14fdd` | `baselines/external/controlgenai_t-lora` | `stabilityai/stable-diffusion-xl-base-1.0` | Official SDXL entrypoints are `train.py` and `inference.py`. The README documents `--resolution=1024` and the shipped `tlora_env.yml` for setup. |
| IntLoRA | `main` | `65a8257a4311e0feab9b33477c9748d13d8ce17b` | `baselines/external/csguoh_intlora` | Stable Diffusion 1.5 weights plus the DreamBooth subject dataset | Official entrypoints are `train_dreambooth_quant.py`, `train_dreambooth_quant.sh`, `evaluation.py`, and `get_results.py`. Keep it appendix-only because the official substrate is Stable Diffusion 1.5 quantization. |
| MPQ-DM | `main` | `2b05dd79c565969d0144a71b782873a7f7060345` | `baselines/external/wlfeng0509_mpq-dm` | LDM `cin256-v2` checkpoint from `ommer-lab.com` | Official flow is `quant_scripts/collect_input_4_calib.py`, `quant_scripts/quantize_ldm_naive.py`, `quant_scripts/train_ourdm.py`, and `quant_scripts/sample_lora_model.py`. Treat it as partial until the manual `ddpm.py` modification is removed or formalized. |

## Wrapper coverage

- `scripts/run_tlora_baseline.py` wraps the official T-LoRA `train.py` entrypoint, validates the registered checkout against `baselines/registry.lock.json`, and writes a deterministic dry-run command under `outputs/baselines/t_lora/<run_name>/`.
- `scripts/run_intlora_baseline.py` wraps the official IntLoRA `train_dreambooth_quant.py` entrypoint only for the official SD1.5 appendix path. It does not attempt an intrusive SDXL port.

## Environment plan

- `environment/rdlora-env.yaml` is the repo-local SDXL-first default for the RD-LoRA substrate.
- The base stack is anchored on official T-LoRA SDXL versions: Python 3.11, PyTorch 2.6.0, torchvision 0.21.0, accelerate 0.26.1, diffusers 0.25.1, transformers 4.37.1, and peft 0.7.0.
- For the exact pinned `huggingface/diffusers` main examples, install the local checkout from `baselines/external/huggingface_diffusers` into that environment and use a disposable overlay if the current text-to-image script requires newer `peft`.
- Do not merge IntLoRA's official Python 3.9 / CUDA 11.8 / Stable Diffusion 1.5 stack into the critical SDXL environment. Keep it appendix-only.

## Unresolved blockers

- `TSM`: no official runnable repository was verified, so it cannot enter the registry or the critical path.
- `AIRA`: no official runnable repository could be verified from an official source on 2026-04-08, so it stays out of the registry despite being relevant conceptually.
- `AccuQuant`: the official project page exposes paper materials but no verified official code repository, so it remains literature-only.
- `IntLoRA` and `MPQ-DM`: both are official appendix baselines, but promoting either to the critical path would force a substrate switch away from SDXL.
