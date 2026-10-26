#!/usr/bin/env bash
# Evaluate the epoch-2500 checkpoint on the validation split with several
# repeated-sample counts. Each count gets its own prediction directory:
# prediction_epoch_2500_samples025/020/015/010/005.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 RUN_DIR DATASET_ROOT [GPU_ID]" >&2
  echo "Example: $0 /data/.../runs/my_run /data/.../dataset 0" >&2
  exit 2
fi

RUN_DIR="$1"
DATASET_ROOT="$2"
GPU_ID="${3:-0}"
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

RUN_DIR="$(cd "$RUN_DIR" && pwd)"
DATASET_ROOT="$(cd "$DATASET_ROOT" && pwd)"
RUNS_ROOT="$(dirname "$RUN_DIR")"
RUN_NAME="$(basename "$RUN_DIR")"
cd "$PROJECT_DIR"

echo "Evaluating $RUN_NAME at epoch 2500 on the validation split"
echo "Dataset: $DATASET_ROOT"
echo "GPU: $GPU_ID"

for num_samples in 25 20 15 10 5; do
  echo "== num_samples=$num_samples =="
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" evaluate_matrix.py \
    --runs_root "$RUNS_ROOT" \
    --dataset_root "$DATASET_ROOT" \
    --device cuda:0 \
    --num_samples "$num_samples" \
    --grid 2500 \
    --only "$RUN_NAME" \
    --grid_only \
    --force \
    --amp \
    --cond_batch 4 \
    --no_summary
  printf -v sample_tag '%03d' "$num_samples"
  cp "$RUN_DIR/metrics.csv" "$RUN_DIR/metrics_epoch_2500_samples${sample_tag}.csv"
done

echo "All evaluations completed. Outputs:"
for num_samples in 25 20 15 10 5; do
  printf '  %s/prediction_epoch_2500_samples%03d\n' "$RUN_DIR" "$num_samples"
done
