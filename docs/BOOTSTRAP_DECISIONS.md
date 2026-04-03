# VeriCodec-Diff pre-coding bootstrap decisions

This pack assumes the locked project scope from the handoff: frozen SANA/DC-AE-family backbone, 1024x1024 generation, 64x64 patch grid, 2x RTX 3090 (24 GB each), and a week-3 hard kill gate.

## Final decisions on the 10 open machine-audit questions

1. **Environment manager**
   - Use **conda environments** as the project standard.
   - Use **mamba** as the solver if available, because it is a drop-in replacement for conda and usually solves faster.
   - Do **not** make Docker part of phase 0.
   - Keep exactly two project envs:
     - `sana-env` for SANA/DC-AE, verifier, refiner, figures, metrics.
     - `ect-env` only for the public ECT baseline.

2. **CUDA mismatch**
   - Do **not** try to "fix" the driver/toolkit mismatch globally.
   - Keep the host driver as-is.
   - For `sana-env`, install **PyTorch 2.4.1 + pytorch-cuda=12.1**.
   - For `ect-env`, match the public ECT env: **PyTorch 2.3.0 + pytorch-cuda=12.1**.
   - Treat the host `nvcc 11.8` as irrelevant unless you later need to build a custom CUDA extension.
   - In phase 0, avoid compiling flash-attn or custom kernels.

3. **Storage layout**
   - Repo code: `~/src/vericodec-diff`
   - Scratch/data/checkpoints/outputs: `/mnt/data/vericodec-diff`
   - Cold archive: `/home/dima/HDD/vericodec-diff-archive`
   - Keep `logs/`, `outputs/`, `checkpoints/` outside git.
   - Important note: the audited machine does **not** meet the earlier 4 TB fast-storage target. Operate in low-storage mode until more NVMe capacity exists.

4. **Editor / IDE**
   - Use **VS Code + integrated terminal**.
   - Use Codex from the terminal inside the repo.
   - Do not make the IDE itself part of the research risk.

5. **GitHub SSH**
   - Configure SSH **before the first push**.
   - Generate an Ed25519 key, add it to the agent, then add the public key to GitHub.

6. **Branching strategy**
   - Keep `main` as the only long-lived branch.
   - Use short-lived task branches only:
     - `bootstrap/...`
     - `infra/...`
     - `exp1/...`
     - `exp2/...`
     - `exp3/...`
     - `exp4/...`
     - `paper/...`
     - `fix/...`
   - Merge back quickly. No `develop` branch.

7. **Docker**
   - Not needed for phase 0.
   - Revisit only after the baseline stack is stable and reproducible in conda.
   - If needed later, use Docker only for archival reproduction, not for day-1 development.

8. **Repo-first or env-first**
   - Create the **local repo skeleton first**, because the environment files, AGENTS.md, skills, and scripts belong in the repo.
   - Configure SSH before creating the remote or at latest before the first push.
   - Then solve envs from the files committed in the repo.

9. **Exact order of operations**
   1. Run preflight audit.
   2. Create storage directories.
   3. Configure GitHub SSH.
   4. Initialize local repo.
   5. Add AGENTS, skills, hooks, env files, bootstrap docs.
   6. Create remote repo and push.
   7. Create `sana-env`.
   8. Run SANA smoke test.
   9. Create `ect-env`.
   10. Freeze baseline registry.
   11. Start week-1 kill-test setup.

10. **What is still missing from the machine/setup information**
   - Whether Docker is installed and whether the NVIDIA Container Toolkit is installed.
   - Whether Node.js/npm or Homebrew are installed for Codex CLI installation.
   - Whether the user has passwordless sudo.
   - Filesystem throughput for `/home`, `/mnt/data`, and HDD.
   - `nvidia-smi topo -m` output and whether GPU P2P is available.
   - Whether any quota / cleanup daemon exists on `/mnt/data`.
   - Whether `tesseract-ocr` is already installed.
   - Exact internet bandwidth for large checkpoint downloads.

## Hard recommendation

Proceed with **VeriCodec-Diff** first. Do not train a refiner until the week-3 sparsity/predictability gate passes.
