#!/bin/bash
set -e  # Stop script immediately if any command fails

# ----------------------------------
# CONFIGURATION
# ----------------------------------
WANDB_API_KEY="YOUR_KEY_HERE"  # Replace with your actual key
# ----------------------------------

echo ">>> [1/5] System Update & Dependencies..."
# RunPod pods are root by default, so no sudo is needed
apt-get update
apt-get install -y git wget nano unzip build-essential tmux htop

echo ">>> [2/5] Creating Virtual Environment (Inheriting PyTorch)..."
# The --system-site-packages flag allows the venv to magically access 
# the massive PyTorch libraries already installed on the RunPod container.
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

echo ">>> [3/5] Detecting Pre-Installed PyTorch & CUDA..."
# Dynamically extract the exact versions from the existing PyTorch installation
PT_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_SUFFIX=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")

WHEEL_URL="https://data.pyg.org/whl/torch-${PT_VER}+${CUDA_SUFFIX}.html"
echo "    -> Detected PyTorch: $PT_VER"
echo "    -> Detected CUDA: $CUDA_SUFFIX"
echo "    -> Using Wheel URL: $WHEEL_URL"

echo ">>> [4/5] Installing PyG and Deep Learning Dependencies..."
# 1. Install the difficult PyG C++ extensions using the dynamically matched wheels
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f $WHEEL_URL

# 2. Install all your required libraries directly (no requirements.txt needed)
pip install torch_geometric scikit-learn scipy tensorboard torchsummary ase e3nn wandb

echo ">>> [5/5] Setting up Directories..."
mkdir -p data results
echo "    -> Directories created. Please manually upload your dataset into the 'data/' folder."

# Log in to WandB non-interactively
export WANDB_API_KEY=$WANDB_API_KEY

echo "========================================================"
echo "Setup complete! Virtual environment is ready."
echo "1. Upload your dataset into the 'data/' folder."
echo "2. Type 'tmux' to start a persistent session."
echo "3. Run: source .venv/bin/activate"
echo "4. Execute your training script."
echo "========================================================"