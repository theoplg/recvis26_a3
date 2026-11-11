#!/bin/bash
#SBATCH --job-name=clip_b32_ensemble
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

cd ~/recvis26_a3
source .venv/bin/activate

echo "host=$(hostname) job=${SLURM_JOB_ID} start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, torch.__file__)"

python clip_zeroshot.py \
    --data data \
    --output runs/clip_b32_ensemble \
    --model ViT-B-32 \
    --pretrained laion2b_s34b_b79k \
    --prompts ensemble \
    --workers 4

echo "end=$(date)"
