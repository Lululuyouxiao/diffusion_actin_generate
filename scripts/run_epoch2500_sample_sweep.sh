#!/usr/bin/env bash
# Evaluate the epoch-2500 checkpoint on the validation split using two GPUs.
# Each repeated-sample count gets its own metrics CSV and prediction directory:
# prediction_epoch_2500_samples025/020/015/010/005.
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

echo "Evaluating $RUN_NAME at epoch 2500 on the validation split"
echo "Dataset: $DATASET_ROOT"
echo "GPU $GPU0_ID: 25, 15 samples"
echo "GPU $GPU1_ID: 20, 10, 5 samples"

run_worker() {
  local gpu_id="$1"
  shift
  local num_samples sample_tag
  for num_samples in "$@"; do
    printf -v sample_tag '%03d' "$num_samples"
    echo "[GPU $gpu_id] num_samples=$num_samples"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" evaluate_matrix.py \
      --runs_root "$RUNS_ROOT" \
      --dataset_root "$DATASET_ROOT" \
      --device cuda:0 \
      --num_samples "$num_samples" \
      --grid 2500 \
      --exact_run "$RUN_NAME" \
      --metrics_filename "metrics_epoch_2500_samples${sample_tag}.csv" \
      --grid_only \
      --force \
      --amp \
      --cond_batch 4 \
      --no_summary
  done
}

LOG_DIR="$RUN_DIR/eval_epoch2500_sample_sweep_logs"
mkdir -p "$LOG_DIR"
run_worker "$GPU0_ID" 25 15 > "$LOG_DIR/gpu${GPU0_ID}.log" 2>&1 &
PID0=$!
run_worker "$GPU1_ID" 20 10 5 > "$LOG_DIR/gpu${GPU1_ID}.log" 2>&1 &
PID1=$!

STATUS0=0
STATUS1=0
wait "$PID0" || STATUS0=$?
wait "$PID1" || STATUS1=$?
if (( STATUS0 != 0 || STATUS1 != 0 )); then
  echo "Evaluation failed (GPU $GPU0_ID: $STATUS0, GPU $GPU1_ID: $STATUS1). Logs: $LOG_DIR" >&2
  exit 1
fi

echo "All evaluations completed. Outputs:"
for num_samples in 25 20 15 10 5; do
  printf '  %s/prediction_epoch_2500_samples%03d\n' "$RUN_DIR" "$num_samples"
done
echo "Logs: $LOG_DIR"
