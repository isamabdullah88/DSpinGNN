#!/bin/bash
set -e  # Stop script immediately if any command fails

# ----------------------------------
WANDB_API_KEY=""  # Or pass it as env var
DATA_DOWNLOAD_ID="1LyzVyRgdE0H2EFlU6xpOpPscO4GSBLWf"  # Drive file ID
# ----------------------------------

echo ">>> [1/6] System Update & Essentials..."
apt-get update && apt-get install -y git wget nano

echo ">>> [2/6] Installing PyTorch with CUDA Support..."
CUDA_VER="12.4"
if [[ $CUDA_VER == "12"* ]]; then CUDA_SUFFIX="cu124"; else CUDA_SUFFIX="cu${CUDA_VER}"; fi

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/$CUDA_SUFFIX

echo ">>> [3/6] Detecting Environment for PyTorch Geometric..."
PT_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
WHEEL_URL="https://data.pyg.org/whl/torch-${PT_VER}+${CUDA_SUFFIX}.html"
echo "    -> Detected PyTorch: $PT_VER, CUDA: $CUDA_SUFFIX"
echo "    -> Using Wheel URL: $WHEEL_URL"

echo ">>> [4/6] Installing Python Dependencies..."
# Install the difficult GNN libraries using the correct wheel URL
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f $WHEEL_URL
pip install torch_geometric e3nn wandb gdown

echo ">>> [5/6] Setting up Data..."
mkdir -p data results

# OPTION A: Download Data (If you have a link)
if [ ! -z "$DATA_DOWNLOAD_ID" ]; then
    echo "Downloading data from URL..."
    cd data
    gdown https://drive.google.com/uc?id="$DATA_DOWNLOAD_ID" -O md17_aspirin.npz
    cd ..
    DATA_PATH="data/md17_aspirin.npz"
fi

echo ">>> [6/6] Starting Training..."
# Log in to WandB non-interactively
export WANDB_API_KEY=$WANDB_API_KEY

# Run the training
# Note: Removed --kaggle True, assuming you adjust defaults for Linux paths
# python train.py \
#     --data_dir "$DATA_PATH" \
#     --batch_size 32 \
#     --epochs 5000 \
#     --WANDB_KEY "$WANDB_API_KEY" \
#     --project "NequIP_Aspirin" \
#     --runname "Aspirin_Test_Run"

# echo ">>> Done! Training Complete."