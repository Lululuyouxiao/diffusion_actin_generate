#!/usr/bin/env bash
# Dual-GPU parallel launcher for evaluate_matrix.py.
# Splits the 7 matrix runs across GPU 0 and GPU 1, writes per-GPU logs.
#
# Usage:
#   bash scripts/run_eval_dual_gpu.sh RUNS_ROOT DATASET_ROOT [NUM_SAMPLES]
#
# Example:
#   bash scripts/run_eval_dual_gpu.sh \
#     /data/yuhanzhao/diffusion_model/runs/matrix_dp2 \
#     /data/yuhanzhao/diffusion_model/Actin-stress-fibers-generator-main/dataset \
#     30
set -uo pipefail

RUNS_ROOT="${1:?Usage: $0 RUNS_ROOT DATASET_ROOT [NUM_SAMPLES]}"
DATASET_ROOT="${2:?Usage: $0 RUNS_ROOT DATASET_ROOT [NUM_SAMPLES]}"
NUM_SAMPLES="${3:-30}"
PYTHON_BIN="${PYTHON_BIN:-python}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$RUNS_ROOT/eval_logs"
mkdir -p "$LOG_DIR"

# Split: 4 runs on GPU0 (which also writes the cross-run summary plot),
#        3 runs on GPU1 (--no_summary to avoid writing the same png).
GPU0_RUNS="paper_baseline,lr_low,timesteps_1200,no_augmentation"
GPU1_RUNS="batch_8,huber_loss,timesteps_400"

echo "[gpu0] runs: $GPU0_RUNS  -> $LOG_DIR/eval_gpu0.log"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" evaluate_matrix.py \
  --runs_root "$RUNS_ROOT" \
  --dataset_root "$DATASET_ROOT" \
  --device cuda:0 \
  --num_samples "$NUM_SAMPLES" \
  --only "$GPU0_RUNS" \
  --amp --cond_batch 4 \
  > "$LOG_DIR/eval_gpu0.log" 2>&1 &
PID0=$!

echo "[gpu1] runs: $GPU1_RUNS  -> $LOG_DIR/eval_gpu1.log"
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" evaluate_matrix.py \
  --runs_root "$RUNS_ROOT" \
  --dataset_root "$DATASET_ROOT" \
  --device cuda:0 \
  --num_samples "$NUM_SAMPLES" \
  --only "$GPU1_RUNS" \
  --no_summary \
  --amp --cond_batch 4 \
  > "$LOG_DIR/eval_gpu1.log" 2>&1 &
PID1=$!

wait "$PID0"; STATUS0=$?
wait "$PID1"; STATUS1=$?

echo "gpu0 exit: $STATUS0"
echo "gpu1 exit: $STATUS1"
if [ "$STATUS0" -eq 0 ] && [ "$STATUS1" -eq 0 ]; then
  echo "ALL DONE. See $RUNS_ROOT/evaluation_summary.png and per-run evaluation_trend.png"
else
  echo "some worker failed; check $LOG_DIR/eval_gpu*.log"
  exit 1
fi
