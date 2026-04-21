#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=Eval_Exodo_GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1              # 1 GPU per array task
#SBATCH --output=/home1/adoyle2025/CLRerNet/local_bash/exps/array_%A_%a.log
#SBATCH --mail-user=adoyle2025@my.fit.edu
#SBATCH --mail-type=END,FAIL

# --- JOB ARRAY SPECIFICATION ---
# 0 is for full.txt, 1-100 are for the Run files
#SBATCH --array=0-100

# --- Environment Setup ---
export CLRERNET_ROOT="/home1/adoyle2025/CLRerNet-Runtime-Monitor-for-Lane-Detection"
export PYTHONPATH=$PYTHONPATH:$CLRERNET_ROOT
export OMP_NUM_THREADS=1

cd $CLRERNET_ROOT
source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh
conda activate clrernet

# Set workspace config for deterministic CuBLAS
export CUBLAS_WORKSPACE_CONFIG=:4096:8

TARGET_DIR=$1

if [ -z "$TARGET_DIR" ]; then
    echo "❌ Error: Provide target directory as arg"
    exit 1
fi

# --- Logic to assign files to IDs ---

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    # Index 0 is strictly dedicated to full.txt
    CURRENT_FILE="$TARGET_DIR/full.txt"
else
    # Indices 1-100 map to Run_1.txt through Run_100.txt
    # We use the ID directly to construct the filename
    CURRENT_FILE="$TARGET_DIR/Run_${SLURM_ARRAY_TASK_ID}.txt"
fi

# --- Execution ---
if [ -f "$CURRENT_FILE" ]; then
    echo "----------------------------------------------------"
    echo "Job ID: $SLURM_ARRAY_JOB_ID | Task ID: $SLURM_ARRAY_TASK_ID"
    echo "Processing: $(basename "$CURRENT_FILE")"
    echo "Host: $(hostname)"
    echo "Time: $(date)"
    echo "----------------------------------------------------"

    # Run your python script (GPU is now visible and requested)
    python /home1/adoyle2025/CLRerNet/local_bash/test_experiment.py "$CURRENT_FILE"
    
    echo "----------------------------------------------------"
    echo "✅ Task $SLURM_ARRAY_TASK_ID Finished at $(date)"
else
    echo "⚠️ File not found: $CURRENT_FILE"
    exit 1
fi