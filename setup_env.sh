#!/usr/bin/env bash
# Create a conda env for NVIDIA's structured-data-models (Kumo Relational, TabICLv2)
# plus XGBoost for baselines. Works on macOS (CPU) and Linux with an NVIDIA GPU.
# Usage: bash setup_env.sh [env_name]
set -euo pipefail

ENV_NAME="${1:-gfm}"

# On a machine with an NVIDIA GPU, request the CUDA builds of PyTorch and XGBoost.
if command -v nvidia-smi >/dev/null 2>&1; then
    TORCH="pytorch>=2.7=cuda*"
    XGB="py-xgboost=*=cuda*"
else
    TORCH="pytorch>=2.7"
    XGB="py-xgboost"
fi

# Install the compiled packages from conda-forge so they share one OpenMP
# runtime (pip wheels of torch/xgboost each bundle their own, which crashes on macOS).
# sdm requires Python >= 3.11 and PyTorch >= 2.7.
conda create -y -n "$ENV_NAME" -c conda-forge \
    python=3.11 "$TORCH" "$XGB" optuna scikit-learn pandas pyarrow py7zr
conda run -n "$ENV_NAME" pip install structured-data-models kaggle

echo "Done. Activate with: conda activate $ENV_NAME"
