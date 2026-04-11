#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --output=test.log
#SBATCH --partition=h200
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --mail-user=adoyle2025@my.fit.edu
#SBATCH --mail-type=END,FAIL

cd /home1/adoyle2025/CLRerNet-Runtime-Monitor-for-Lane-Detection

export PYTHONPATH=$PYTHONPATH:$(pwd)

source /home1/adoyle2025/miniconda3/etc/profile.d/conda.sh

conda activate clrernet

python tools/calculate_frame_diff.py "/home1/adoyle2025/Datasets/Datasets/CULane"

python tools/train.py configs/clrernet/culane/clrernet_culane_dla34.py
