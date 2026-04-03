# Week-1 order of operations

1. Run `bootstrap/00_preflight.sh`.
2. Initialize repo with `bootstrap/01_repo_init.sh`.
3. Configure GitHub SSH.
4. Push the bootstrap skeleton to GitHub.
5. Create `sana-env` and `ect-env` with `bootstrap/03_env_create.sh`.
6. Run `python bootstrap/04_sana_smoke_test.py` inside `sana-env`.
7. Clone baseline repos and write `baselines/registry.lock.json`.
8. Freeze prompt manifest skeletons.
9. Implement the four week-1 scripts only:
   - `check_environment.py`
   - `generate_diagnostic_dataset.py`
   - `generate_sana_outputs.py`
   - `compute_patch_errors.py`
10. Do not start refiner work.
