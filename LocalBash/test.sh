#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=CLRerNET
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --output=test.log
#SBATCH --partition=h200
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-user=adoyle2025@my.fit.edu
#SBATCH --mail-type=END,FAIL

cd /home1/adoyle2025/CLRerNet-Runtime-Monitor-for-Lane-Detection

# Set PYTHONPATH to include your local project
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Activate environment
source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh
conda activate clrernet

# Set workspace config for deterministic CuBLAS operations
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Use torchrun to launch the distributed training
# --nproc_per_node=4 tells torchrun to use 4 GPUs on this node
# --nnodes=1 defines the number of nodes
# --rdzv_endpoint=localhost:29500 handles the distributed synchronization
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --rdzv_id=job_224863 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    tools/train.py configs/clrernet/culane/clrernet_culane_dla34_ema.py \
    --launcher pytorch
