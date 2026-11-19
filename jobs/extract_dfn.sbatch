#!/bin/bash
#SBATCH --job-name=extract_dfn
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# Safe to resubmit: finished chunks and splits are skipped.

set -euo pipefail

cd ~/recvis26_a3
source .venv/bin/activate

echo "host=$(hostname) job=${SLURM_JOB_ID} start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python extract_features.py \
    --data data \
    --output features/vith_dfn5b \
    --model ViT-H-14-378-quickgelu \
    --pretrained dfn5b \
    --batch-size 64 \
    --chunk-size 2048 \
    --workers 4 \
    --fp16

echo "end=$(date)"