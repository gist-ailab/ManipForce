#!/bin/bash
# Ablation experiments for diffusion step redundancy paper
# GPU: single RTX 3090
# Each experiment ~500 epochs

set -eo pipefail  # exit on error, including pipe failures

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/ablations_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "========================================="
echo " Ablation Experiments"
echo " Started: $(date)"
echo " Logs: ${LOG_DIR}/"
echo "========================================="

# 1. Linear schedule (cosine vs linear ablation)
echo ""
echo "[1/3] Linear schedule experiment..."
echo "      Start: $(date)"
python scripts/train.py \
    --config-name=manipforce_ods3_256x256_linear \
    exp_name=linear_schedule \
    logging.name=linear_schedule \
    2>&1 | tee "${LOG_DIR}/01_linear_schedule.log"
echo "      Done: $(date)"

# 2. Noisy regression (noise O, timestep X)
echo ""
echo "[2/3] Noisy regression experiment..."
echo "      Start: $(date)"
python scripts/train.py \
    --config-name=manipforce_ods3_256x256_noisy_regression \
    exp_name=noisy_regression \
    logging.name=noisy_regression \
    2>&1 | tee "${LOG_DIR}/02_noisy_regression.log"
echo "      Done: $(date)"

# 3. Pure regression (noise X, timestep X)
echo ""
echo "[3/3] Pure regression experiment..."
echo "      Start: $(date)"
python scripts/train.py \
    --config-name=manipforce_ods3_256x256_regression \
    exp_name=pure_regression \
    logging.name=pure_regression \
    2>&1 | tee "${LOG_DIR}/03_pure_regression.log"
echo "      Done: $(date)"

echo ""
echo "========================================="
echo " All experiments finished: $(date)"
echo " Logs saved to: ${LOG_DIR}/"
echo "========================================="
