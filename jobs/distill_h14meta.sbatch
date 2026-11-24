#!/bin/bash
#SBATCH --job-name=distill_h14meta
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%j.out

# Round 3: bigger student (ViT-H/14 DFN-5B, 224 px), best teacher so far (ensemble of 2 students,
# 97.30 % public), longer schedule, RandAugment + Mixup/CutMix with mixed teacher targets.
# Safe to resubmit: training resumes from runs/distill_h14meta/checkpoint.pt

set -euo pipefail

cd ~/recvis26_a3
source .venv/bin/activate

echo "host=$(hostname) job=${SLURM_JOB_ID} start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python distill.py \
    --data data \
    --teacher runs/h14_knn_bal/test_probs.npy \
    --output runs/distill_h14meta \
    --model ViT-H-14-quickgelu \
    --pretrained metaclip_fullcc \
    --epochs 8 \
    --batch-size 64 \
    --workers 4 \
    --grad-checkpointing \
    --randaugment \
    --mixup 0.8 \
    --cutmix 1.0

echo "end=$(date)"
