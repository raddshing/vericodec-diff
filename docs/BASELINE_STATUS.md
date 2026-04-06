# Baseline status

Status definitions:

- `available`: official code checkout is present locally and upstream documents a concrete run path.
- `partial`: official checkout exists, but key code or run instructions are incomplete.
- `missing`: no official usable checkout was found.
- `use_proxy`: official checkout exists, but VeriCodec-Diff should execute through a different local baseline until upstream releases runnable code.

| Baseline | Upstream | Branch | Pinned commit | Local checkout | Status | Expected weights / checkpoints | Entry points / launch notes | Environment notes | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D2iT | `https://github.com/jiawn-creator/Dynamic-DiT.git` | `main` | `1c3973c7d8492693723980fc3f8a7e9b83d4c5e7` | `baselines/external/jiawn-creator_dynamic-dit` | `use_proxy` | None published in the official checkout. | No runnable entrypoints in the official checkout. Execute via proxy at `baselines/external/microsoft_ras`. | No environment file or setup instructions are present in the official checkout. | Official checkout contains only `README.md` with `Coming soon!` and no code, setup files, scripts, or released weights. |
| ECT | `https://github.com/locuslab/ect.git` | `main` | `4311059770f54821d151a9b0e1f76770a5f3930e` | `baselines/external/locuslab_ect` | `available` | `https://drive.google.com/file/d/1WN_eLTrcl-vB7fMc1HADpacgcO4SNJ_1/view?usp=sharing` | `run_ecm_1hour.sh`, `run_ecm.sh`, `eval_ecm.sh` | Upstream `env.yml` targets Python 3.9.18 / PyTorch 2.3.0. This pinned checkout is `main` for CIFAR-10; upstream also documents an `imgnet` branch for ImageNet 64x64. Repo-local `environment/ect-env.yaml` is the closest tracked match. |  |
| EfficientViT / DC-AE | `https://github.com/mit-han-lab/efficientvit.git` | `master` | `de7d7733cc0329f391b33f1f459271562ec27bd5` | `baselines/external/mit-han-lab_efficientvit` | `available` | `mit-han-lab/dc-ae-f32c32-in-1.0-diffusers`, `mit-han-lab/dc-ae-f64c128-in-1.0-diffusers`, `mit-han-lab/dc-ae-f128c512-in-1.0-diffusers` | `applications/dc_ae/demo_dc_ae_model_diffusers.py`, `applications/dc_ae/demo_dc_ae_diffusion_model.py`, `applications/dc_ae/eval_dc_ae_model.py`, `applications/dc_ae/eval_dc_ae_diffusion_model.py`, `applications/dc_ae/train_dc_ae_diffusion_model.py` | Upstream setup is `conda create -n efficientvit python=3.10` then `pip install -U -r requirements.txt`. No dedicated repo-local environment YAML exists for this stack. |  |
| RAS | `https://github.com/microsoft/RAS.git` | `main` | `a9c5ec8fb2147ef450a7e3b5ec3b7bdd0006de3f` | `baselines/external/microsoft_ras` | `available` | `stabilityai/stable-diffusion-3-medium-diffusers`, `Alpha-VLLM/Lumina-Next-SFT-diffusers` | `scripts/Stable_Diffusion_3_example.sh`, `scripts/Lumina_Next_T2I_example.sh` | Upstream setup is `conda create -n ras python=3.12`, `python setup.py install`, `pip install flash-attn --no-build-isolation`. The README’s SD3 example uses `use_auth_token=True`. `PyCuda` is only needed for index fusion. |  |
| Sana | `https://github.com/NVlabs/Sana.git` | `main` | `0e62d975d5e1357226131f78d15b544f94875b76` | `baselines/external/nvlabs_sana` | `available` | `Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers` | `environment_setup.sh`, `app/app_sana.py`. Upstream quick start also documents `SanaPipeline.from_pretrained("Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers")`. | Repo-local `environment/sana-env.yaml` already includes the `diffusers` support needed for `SanaPipeline`. Upstream quick start bootstraps with `./environment_setup.sh sana`. |  |

## D2iT fallback

The official D2iT repository exists locally at `baselines/external/jiawn-creator_dynamic-dit`, but it is still a placeholder checkout with only a README and no runnable code. Until upstream publishes an actual implementation, VeriCodec-Diff should treat `baselines/external/microsoft_ras` as the execution proxy for the D2iT slot in competitor tracking.
