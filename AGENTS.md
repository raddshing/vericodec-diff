# VeriCodec-Diff project instructions

## Mission
Build and evaluate **VeriCodec-Diff** first.
The locked first-paper claim is: a frozen deeply compressed latent diffusion backbone, an inference-time verifier that predicts patchwise compression failure, and sparse repair only on flagged regions.

## Non-negotiable constraints
- Backbone stays frozen unless the task explicitly says otherwise.
- Default image size: 1024x1024.
- Default patch size: 64x64 -> 16x16 grid.
- Do not begin refiner training until the week-3 kill gate passes.
- Every runnable script must save `resolved_config.yaml` before doing work.
- Paths must be deterministic and stable.

## Project phases
1. Bootstrap and smoke tests.
2. Week-1 to week-3 kill test.
3. Exp1 failure localization.
4. Exp2 sparse repair.
5. Exp3 competitor comparison.
6. Exp4 temporal-vs-spatial orthogonality.
7. Journal ablations.

## Coding rules
- Prefer small importable modules under `src/` and thin entrypoints under `scripts/`.
- Use argparse + OmegaConf, not Hydra runtime composition.
- Never hardcode absolute paths except through config defaults.
- Write JSON/CSV metrics for every experiment.
- Add at least one smoke test when introducing a new pipeline component.

## Run discipline
- Start with the smallest slice that can fail fast.
- When changing evaluation logic, rerun the corresponding smoke test first.
- When touching metrics or figure code, do not recompute model outputs unless necessary.
- Keep notes in `outputs/memos/` for major decisions and deviations.

## Repo hygiene
- Do not commit checkpoints, raw data, or generated outputs.
- Record third-party repo status in `baselines/registry.lock.json`.
- Keep `main` releasable; use short-lived branches for experiments.
