#!/usr/bin/env bash
# Evaluate epoch 1500 for four selected runs on two GPUs, using 100 samples.
# Each worker writes only its assigned runs. No best.pt checkpoint is loaded
# or sampled, and the cross-run summary is intentionally left for a later pass.
set -uo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 RUNS_ROOT DATASET_ROOT" >&2
  exit 2
fi

RUNS_ROOT="$1"
DATASET_ROOT="$2"
NUM_SAMPLES=100
EPOCH=1500
PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$RUNS_ROOT/eval_epoch1500_100samples_logs"
mkdir -p "$LOG_DIR"
cd "$PROJECT_DIR"

GPU0_RUNS="lr_low,paper_baseline"
GPU1_RUNS="timesteps_1200,huber_loss"

shopt -s nullglob
for group in lr_low paper_baseline timesteps_1200 huber_loss; do
  found=0
  for run_dir in "$RUNS_ROOT"/*"$group"*; do
    [[ -d "$run_dir" ]] || continue
    found=1
    if [[ ! -f "$run_dir/metrics.csv" || ! -f "$run_dir/checkpoints/epoch_${EPOCH}.pt" ]]; then
      echo "Missing metrics.csv or epoch_${EPOCH}.pt in $run_dir" >&2
      exit 1
    fi
  done
  if [[ "$found" -eq 0 ]]; then
    echo "No run directory matching '$group' under $RUNS_ROOT" >&2
    exit 1
  fi
done

echo "[gpu0] $GPU0_RUNS -> $LOG_DIR/gpu0.log"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" evaluate_matrix.py \
  --runs_root "$RUNS_ROOT" \
  --dataset_root "$DATASET_ROOT" \
  --device cuda:0 \
  --num_samples "$NUM_SAMPLES" \
  --grid "$EPOCH" \
  --grid_only \
  --only "$GPU0_RUNS" \
  --force \
  --no_summary \
  --amp --cond_batch 4 \
  > "$LOG_DIR/gpu0.log" 2>&1 &
PID0=$!

echo "[gpu1] $GPU1_RUNS -> $LOG_DIR/gpu1.log"
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" evaluate_matrix.py \
  --runs_root "$RUNS_ROOT" \
  --dataset_root "$DATASET_ROOT" \
  --device cuda:0 \
  --num_samples "$NUM_SAMPLES" \
  --grid "$EPOCH" \
  --grid_only \
  --only "$GPU1_RUNS" \
  --force \
  --no_summary \
  --amp --cond_batch 4 \
  > "$LOG_DIR/gpu1.log" 2>&1 &
PID1=$!

wait "$PID0"; STATUS0=$?
wait "$PID1"; STATUS1=$?

echo "gpu0 exit: $STATUS0"
echo "gpu1 exit: $STATUS1"
if [[ "$STATUS0" -eq 0 && "$STATUS1" -eq 0 ]]; then
  echo "ALL DONE. Epoch 1500 metrics updated for the four selected runs."
  echo "Logs: $LOG_DIR/gpu0.log and $LOG_DIR/gpu1.log"
else
  echo "Evaluation failed; inspect $LOG_DIR/gpu0.log and $LOG_DIR/gpu1.log" >&2
  exit 1
fi
