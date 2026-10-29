#!/usr/bin/env bash
# Evaluate the epoch-2500 checkpoint on the full paired dataset using two GPUs.
# The single evaluation uses num_samples=25 and DataParallel across both GPUs.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 RUN_DIR DATASET_ROOT [GPU0_ID [GPU1_ID]]" >&2
  echo "Example: $0 /data/.../runs/my_run /data/.../dataset 0 1" >&2
  exit 2
fi

RUN_DIR="$1"
DATASET_ROOT="$2"
GPU0_ID="${3:-0}"
GPU1_ID="${4:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="$RUN_DIR/checkpoints/epoch_2500.pt"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Missing run directory: $RUN_DIR" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing epoch-2500 checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$RUN_DIR/metrics.csv" ]]; then
  echo "Missing training metrics: $RUN_DIR/metrics.csv" >&2
  exit 1
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Missing dataset directory: $DATASET_ROOT" >&2
  exit 1
fi
if [[ "$GPU0_ID" == "$GPU1_ID" ]]; then
  echo "GPU IDs must be different" >&2
  exit 2
fi

RUN_DIR="$(cd "$RUN_DIR" && pwd)"
DATASET_ROOT="$(cd "$DATASET_ROOT" && pwd)"
RUNS_ROOT="$(dirname "$RUN_DIR")"
RUN_NAME="$(basename "$RUN_DIR")"
cd "$PROJECT_DIR"

echo "Evaluating $RUN_NAME at epoch 2500 on the full paired dataset"
echo "Dataset: $DATASET_ROOT"
echo "DataParallel GPUs: $GPU0_ID,$GPU1_ID"
echo "num_samples: 25"

LOG_DIR="$RUN_DIR/eval_epoch2500_sample_sweep_logs"
mkdir -p "$LOG_DIR"
if CUDA_VISIBLE_DEVICES="$GPU0_ID,$GPU1_ID" "$PYTHON_BIN" evaluate_matrix.py \
  --runs_root "$RUNS_ROOT" \
  --dataset_root "$DATASET_ROOT" \
  --device cuda:0 \
  --num_samples 25 \
  --grid 2500 \
  --exact_run "$RUN_NAME" \
  --metrics_filename metrics_epoch_2500_samples025.csv \
  --grid_only \
  --all_data \
  --use_data_parallel \
  --gpu_ids 0,1 \
  --force \
  --amp \
  --cond_batch 4 \
  --no_summary \
  > "$LOG_DIR/gpu${GPU0_ID}_${GPU1_ID}.log" 2>&1; then
  STATUS=0
else
  STATUS=$?
fi
if (( STATUS != 0 )); then
  echo "Evaluation failed (status $STATUS). Log: $LOG_DIR/gpu${GPU0_ID}_${GPU1_ID}.log" >&2
  exit 1
fi

echo "All evaluations completed. Outputs:"
printf '  %s/prediction_epoch_2500_samples025\n' "$RUN_DIR"
echo "Logs: $LOG_DIR"
