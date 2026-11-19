#!/bin/bash
#SBATCH --job-name=distill_l14_r2
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# Safe to resubmit: training resumes from runs/distill_l14_r2/checkpoint.pt

set -euo pipefail

cd ~/recvis26_a3
source .venv/bin/activate

echo "host=$(hostname) job=${SLURM_JOB_ID} start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python distill.py \
    --data data \
    --teacher runs/ens4_distill_knn_bal/test_probs.npy \
    --output runs/distill_l14_r2 \
    --model ViT-L-14 \
    --pretrained laion2b_s32b_b82k \
    --epochs 4 \
    --batch-size 64 \
    --workers 4

echo "end=$(date)"
