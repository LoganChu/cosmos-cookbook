#!/usr/bin/env bash
# Create .venv and install the Cosmos 3 inference stack.
# TORCH_BACKEND must match the CUDA version your driver supports:
#   cu130 for driver CUDA 13.x (driver 580+), cu128 for CUDA 12.x.
set -euo pipefail
cd "$(dirname "$0")/.."
TORCH_BACKEND="${TORCH_BACKEND:-cu130}"
uv venv .venv --python 3.13 --seed --managed-python --allow-existing
uv pip install --python .venv/bin/python --torch-backend="$TORCH_BACKEND" -r requirements.txt
.venv/bin/python scripts/check_env.py
