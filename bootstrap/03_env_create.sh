#!/usr/bin/env bash
set -euo pipefail

SOLVER="conda"
if command -v mamba >/dev/null 2>&1; then
  SOLVER="mamba"
fi

$SOLVER env create -f environment/sana-env.yaml || $SOLVER env update -f environment/sana-env.yaml --prune
$SOLVER env create -f environment/ect-env.yaml || $SOLVER env update -f environment/ect-env.yaml --prune

echo "Environments ready."
echo "Run: conda activate sana-env"
