# TurboContext baseline status

Audit date: 2026-04-08

## Status definitions

- `available`: official code checkout is present locally and upstream documents a concrete run path.
- `partial`: official checkout exists, but upstream run instructions require manual correction or additional local patching before a clean run.
- `literature_only`: no official runnable repository was verified on 2026-04-08.

## Checked baseline repos

| Baseline | Upstream | Branch | Pinned commit | Local checkout | Runnable status | Expected model / checkpoint dependencies | Entrypoint notes | Critical reproduction path | Reason if not on critical path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EasyRef | `https://github.com/TempleX98/EasyRef.git` | `main` | `c8aa6b7703eafcf4ea5106494f3c61bcee6eb62a` | `baselines/external/templex98_easyref` | `available` | `zongzhuofan/EasyRef`, `stabilityai/stable-diffusion-xl-base-1.0`, `Qwen/Qwen2-VL-2B-Instruct` | Inference is notebook-first via `easyref_demo.ipynb`. Training is driven by `scripts/alignment_pretraining.sh`, `scripts/single_ref_finetuning.sh`, and `scripts/multi_ref_finetuning.sh`, all targeting 1024 resolution and large multi-GPU jobs. | No | EasyRef is a multimodal reference-conditioned personalization system on SDXL plus Qwen, not a compressed latent diffusion verifier-and-repair baseline. |
| CacheQuant | `https://github.com/BienLuky/CacheQuant.git` | `main` | `9470ddf1bac09337ce3281f797357ac35997a358` | `baselines/external/bienluky_cachequant` | `partial` | LDM `cin256-v2` checkpoint under `mainldm/models/ldm`, Stable Diffusion weights per CompVis, original evaluation datasets | Core launch scripts are under `mainldm/`, for example `sample_cachequant_imagenet_cali.py`, `sample_cachequant_imagenet_predadd.py`, `sample_cachequant_imagenet_params.py`, and `sample_cachequant_imagenet_quant.py`. The README still points to `err_add/...` even though the checkout uses `error_dec/...`, so the documented path needs manual correction. | No | CacheQuant is a whole-model caching and quantization accelerator. It does not reproduce the frozen compressed backbone plus patchwise verifier plus sparse repair substrate we need for VeriCodec-Diff. |
| DreamCache | `https://github.com/Emanuele97x/DreamCache.git` | `main` | `a0e33b333f7d448d56018a450c4dd99610e85142` | `baselines/external/emanuele97x_dreamcache` | `partial` | Stable Diffusion 2.1 checkpoint `v2-1_512-ema-pruned.ckpt`, DreamCache adapter weights, DreamCache synthetic dataset, DreamBench checkout | Training and eval scripts exist under `scripts/`, including `train_multi_masked.sh`, `sample_ref.sh`, `generate_dreambooth_mult.sh`, `evaluate_dreambooth.sh`, and `gradio_app.py`. The README references `environment.yaml`, but the checkout ships `environment_dreamcache.yml`; the shell scripts also contain placeholder paths that must be edited locally. | No | DreamCache is a personalization method based on feature caching and adapter training, not a patch failure detector or sparse repair system. |

## Extra substrate notes

| Item | Classification | Notes |
| --- | --- | --- |
| Q&C | `literature_only` | No official repository was verified on 2026-04-08 via GitHub and web search, so it should stay out of the runnable baseline set until an official code release is confirmed. |
| SQuat | `component_inspiration` | Track it only for reusable quantization ideas. Do not treat it as a full-system baseline for TurboContext or VeriCodec-Diff competitor runs. |

## Environment file added

`environment/easyref-env.yaml` is a repo-local conda environment for EasyRef. It mirrors the upstream `requirements.txt` and adds `git-lfs`, notebook support, and `deepspeed` so the local checkout has a documented install target without downloading any model weights.
