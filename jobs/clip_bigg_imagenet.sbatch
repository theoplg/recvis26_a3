#!/bin/bash
#SBATCH --job-name=clip_bigg_imagenet
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

cd ~/recvis26_a3
source .venv/bin/activate

echo "host=$(hostname) job=${SLURM_JOB_ID} start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python clip_zeroshot.py \
    --data data \
    --output runs/clip_bigg_openai_imagenet \
    --model ViT-bigG-14 \
    --pretrained laion2b_s39b_b160k \
    --prompts openai_imagenet \
    --batch-size 64 \
    --workers 4 \
    --fp16

echo "end=$(date)"