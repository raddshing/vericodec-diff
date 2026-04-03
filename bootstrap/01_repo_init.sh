#!/usr/bin/env bash
set -euo pipefail

export VERICODEC_REPO="${VERICODEC_REPO:-$HOME/src/vericodec-diff}"
export VERICODEC_DATA="${VERICODEC_DATA:-/mnt/data/vericodec-diff}"
export VERICODEC_ARCHIVE="${VERICODEC_ARCHIVE:-/home/dima/HDD/vericodec-diff-archive}"

mkdir -p "$HOME/src"
mkdir -p "$VERICODEC_REPO"
mkdir -p "$VERICODEC_DATA"/{data,outputs,checkpoints,logs,baselines,experiments,figures}
mkdir -p "$VERICODEC_ARCHIVE"

cd "$VERICODEC_REPO"

if [ ! -d .git ]; then
  git init -b main
fi

mkdir -p environment bootstrap docs .agents/skills .githooks
PACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cp -n "$PACK_ROOT"/environment/*.yaml environment/ || true
cp -n "$PACK_ROOT"/bootstrap/00_preflight.sh bootstrap/ || true
cp -n "$PACK_ROOT"/docs/BOOTSTRAP_DECISIONS.md docs/ || true
cp -n "$PACK_ROOT"/AGENTS.md . || true
cp -rn "$PACK_ROOT"/.agents/skills/* .agents/skills/ || true
cp -n "$PACK_ROOT"/.githooks/pre-commit .githooks/ || true

cat > .gitignore <<'GITEOF'
__pycache__/
*.pyc
*.pyo
*.pyd
.pytest_cache/
.mypy_cache/
.ruff_cache/
.ipynb_checkpoints/
.env
.venv/
logs/
outputs/
checkpoints/
baselines/external/
data/raw/
data/processed/
figures/final/
paper/build/
*.pt
*.pth
*.ckpt
*.safetensors
*.npz
*.npy
*.tar
*.zip
GITEOF

git config core.hooksPath .githooks

echo "Repo initialized at: $VERICODEC_REPO"
echo "Data root: $VERICODEC_DATA"
echo "Archive root: $VERICODEC_ARCHIVE"
echo
echo "Next: configure SSH, create the remote, then solve sana-env." 
