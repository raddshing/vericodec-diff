#!/usr/bin/env bash
set -euo pipefail

echo "== System =="
uname -a || true
lsb_release -a || true

echo "== GPUs =="
nvidia-smi || true
nvidia-smi topo -m || true

echo "== CUDA toolchain =="
which nvcc || true
nvcc --version || true

echo "== Storage =="
df -h /home /mnt/data /home/dima/HDD || true

echo "== Python / conda =="
which python || true
python --version || true
which conda || true
conda --version || true

if command -v mamba >/dev/null 2>&1; then
  echo "mamba: $(mamba --version)"
else
  echo "mamba: not installed"
fi

echo "== Node / npm (for Codex CLI) =="
which node || true
node --version || true
which npm || true
npm --version || true

echo "== Git =="
git --version || true
git config --global user.name || true
git config --global user.email || true

echo "== Docker =="
which docker || true
docker --version || true

echo "== OCR =="
which tesseract || true
tesseract --version || true

echo "== Suggested next command =="
echo "bash bootstrap/01_repo_init.sh"
