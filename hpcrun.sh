#!/bin/bash
#--------------------
# SLURM Parameters
#--------------------
#SBATCH --job-name=V100_AITest # Job name
#SBATCH --output=gpu_test_%j.out # Standard output and error log
#SBATCH --time=00:10:00 # Time limit (10 minutes)
#SBATCH --mem=16G # Memory limit
#SBATCH --partition=hpcshort # IMPORTANT: Change this to your actual GPU partition name
#SBATCH --gres=gpu:v100:1 # Request 1 V100 card (Change to gpu:v100:2 to test both)

echo "--- Starting SLURM Job ---"
echo "Job running on node: $SLURM_JOB_NODELIST"
echo "Visible GPU IDs: $SLURM_VISIBLE_DEVICES"
#--------------------
# Module and Environment Setup
#--------------------
# 1. Load the Miniforge environment module
module load Miniforge/25.3.1
# 2. Enable Conda Shell Functions (CRITICAL)
. /opt/miniforge3/etc/profile.d/conda.sh
# 3. Activate the PyTorch environment
conda activate ai_test_env
if [ $? -ne 0 ]; then
 echo "ERROR: Failed to activate Conda environment 'ai_test_env'."
 exit 1
fi
#--------------------
# Execution
#--------------------
echo "Running Python test..."
python gpu_test.py
#--------------------
# Cleanup
#--------------------
conda deactivate
echo "Job finished."