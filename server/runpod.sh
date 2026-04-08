#!/bin/bash
set -e  # Stop script immediately if any command fails

# ----------------------------------
# CONFIGURATION
# ----------------------------------
WANDB_API_KEY="YOUR_KEY_HERE"  # Replace with your actual key
# ----------------------------------

echo ">>> [1/4] System Update & Dependencies..."
# RunPod pods are root by default, so no sudo is needed
apt-get update
apt-get install -y git wget nano unzip build-essential tmux htop

echo ">>> [2/4] Detecting Pre-Installed PyTorch & CUDA..."
# Dynamically extract the exact versions from the existing system PyTorch installation
PT_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_SUFFIX=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))")

WHEEL_URL="https://data.pyg.org/whl/torch-${PT_VER}+${CUDA_SUFFIX}.html"
echo "    -> Detected PyTorch: $PT_VER"
echo "    -> Detected CUDA: $CUDA_SUFFIX"
echo "    -> Using Wheel URL: $WHEEL_URL"

echo ">>> [3/4] Installing PyG and Deep Learning Dependencies System-Wide..."
# 1. Install the PyG C++ extensions directly to the system using dynamically matched wheels
# (Note: Kept your original list, but remember you can remove cluster/sparse/spline if you don't use them)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f $WHEEL_URL

# 2. Install all your required libraries directly to the system
pip install torch_geometric scikit-learn scipy tensorboard torchsummary ase e3nn wandb

echo ">>> [4/4] Setting up Directories..."
mkdir -p data results
echo "    -> Directories created. Please manually upload your dataset into the 'data/' folder."

# Log in to WandB non-interactively
export WANDB_API_KEY=$WANDB_API_KEY

echo "========================================================"
echo "Setup complete! Libraries installed system-wide."
echo "1. Upload your dataset into the 'data/' folder."
echo "2. Type 'tmux' to start a persistent session."
echo "3. Execute your training script (no environment activation needed)."
echo "========================================================"