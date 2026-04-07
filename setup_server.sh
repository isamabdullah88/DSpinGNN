#!/bin/bash
set -e  # Stop script immediately if any command fails

# ----------------------------------
# CONFIGURATION
# ----------------------------------
WANDB_API_KEY="YOUR_KEY_HERE"  # Replace with your actual key
CUDA_SUFFIX="cu124"
# ----------------------------------

echo ">>> [1/6] System Update & Dependencies..."
sudo apt-get update
# Added software-properties-common to allow adding external PPAs securely
sudo apt-get install -y software-properties-common git wget nano unzip build-essential tmux htop

echo ">>> [2/6] Installing Python 3.12..."
# Add the deadsnakes PPA to guarantee Python 3.12 is available
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
# Install Python 3.12 along with its specific venv and dev (header) packages
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

echo ">>> [3/6] Creating Python 3.12 Virtual Environment..."
# Explicitly call python3.12 to create the environment
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

echo ">>> [4/6] Installing PyTorch with CUDA Support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$CUDA_SUFFIX

echo ">>> [5/6] Detecting Environment for PyTorch Geometric..."
# This will now correctly detect 3.12
PT_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
WHEEL_URL="https://data.pyg.org/whl/torch-${PT_VER}+${CUDA_SUFFIX}.html"
echo "    -> Detected PyTorch: $PT_VER, CUDA: $CUDA_SUFFIX"
echo "    -> Using Wheel URL: $WHEEL_URL"

echo ">>> [6/6] Installing PyG and Deep Learning Dependencies..."
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f $WHEEL_URL
pip install torch_geometric e3nn wandb

# Install the rest of your standard requirements safely
pip install -r requirements.txt

echo ">>> Setting up Directories..."
mkdir -p data results
echo "    -> Directories created. Please manually upload your dataset into the 'data/' folder using VS Code."

# Log in to WandB non-interactively
export WANDB_API_KEY=$WANDB_API_KEY

echo "========================================================"
echo "Setup complete! Python 3.12 Virtual environment is ready."
echo "1. Upload your dataset into the 'data/' folder."
echo "2. Type 'tmux' to start a persistent session."
echo "3. Run: source .venv/bin/activate"
echo "4. Execute your training script."
echo "========================================================"