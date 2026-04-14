#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=Eval_Exodo_A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/home1/adoyle2025/CLRerNet-Runtime-Monitor-for-Lane-Detection/LocalBash/eval_exodo_gpu2_%j.log
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --mail-user=adoyle2025@my.fit.edu
#SBATCH --mail-type=END,FAIL

# --- Environment Setup ---

export CLRERNET_ROOT="/home1/adoyle2025/CLRerNet-Runtime-Monitor-for-Lane-Detection"

# 3. Add BOTH to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$CLRERNET_ROOT

cd $CLRERNET_ROOT

source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh
conda activate clrernet

# Set workspace config for deterministic CuBLAS (matching your training style)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# --- Execution ---
echo "----------------------------------------------------"
echo "Starting Evaluation on A100 (gpu2)"
echo "Target: All JSONs in /home1/adoyle2025/Distribution-Shift-Lane-Perception/LocalBash"
echo "Host: $(hostname)"
echo "----------------------------------------------------"

# Run the aggregation and lane detection script
python tools/eval_batches.py \
        configs/clrernet/culane/clrernet_culane_dla34_ema.py \
        clrernet_culane_dla34_ema.pth \
        --data-root /home1/adoyle2025/Datasets/Datasets/CULane \
        --logs-dir  /home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/Exodo \
        --out-csv   /home1/adoyle2025/Distribution-Shift-Lane-Perception/logs/Exodo/master_evaluation_results.csv

echo "Evaluation finished at: $(date)"
