#!/usr/bin/env bash
# run.sh - run the full pipeline scripts in order
# This script is used by the Docker image and local testing to execute the
# main pipeline stages in sequence.

set -euo pipefail

echo "========================================================================"
echo "[run.sh] Starting full pipeline run at $(date --iso-8601=seconds)"
echo "========================================================================"
echo ""

# Step 1: Data Preprocessing
echo "========================================================================"
echo "[STEP 1/4] Data Preprocessing"
echo "========================================================================"
python src/01-data-preprocessing.py
echo ""

# Step 2: Training (using best config: textcnn_micro with CORAL)
echo "========================================================================"
echo "[STEP 2/4] Model Training"
echo "========================================================================"
python src/02-training.py --model textcnn --coral --embedding-dim 8 --epochs 30 --name textcnn_micro_final
echo ""

# Step 3: Evaluation
echo "========================================================================"
echo "[STEP 3/4] Model Evaluation"
echo "========================================================================"
python src/03-evaluation.py
echo ""

# Step 4: Inference Demo
echo "========================================================================"
echo "[STEP 4/4] Inference Demo"
echo "========================================================================"
python src/04-inference.py
echo ""

echo "========================================================================"
echo "[run.sh] Pipeline finished successfully at $(date --iso-8601=seconds)"
echo "========================================================================"
