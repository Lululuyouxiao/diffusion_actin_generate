#!/usr/bin/env bash
# Resume the existing lr_low run from epoch 1500 to 1800, then evaluate
# checkpoints 1550..1800 on the full validation split.
set -euo pipefail

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [RUNS_ROOT [DATASET_ROOT]]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="${1:-/data/yuhanzhao/diffusion_model/runs/matrix_dp2}"
DATASET_ROOT="${2:-/data/yuhanzhao/diffusion_model/Actin-stress-fibers-generator-main/Actin-stress-fibers-generator-main/dataset}"
RUN_NAME="lr_low__ep1500_bs16_lr7.5e-05_T800_S80_loss-mse_aug1_dp-0-1"
RUN_DIR="$RUNS_ROOT/$RUN_NAME"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME_CHECKPOINT="$RUN_DIR/last.pt"
GRID="1550,1600,1650,1700,1750,1800"

cd "$PROJECT_DIR"

if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "Missing resume checkpoint: $RESUME_CHECKPOINT" >&2
  exit 1
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Missing dataset directory: $DATASET_ROOT" >&2
  exit 1
fi

RESUME_EPOCH="$($PYTHON_BIN -c 'import sys, torch; state = torch.load(sys.argv[1], map_location="cpu", weights_only=False); print(int(state["epoch"]))' "$RESUME_CHECKPOINT")"
if [[ "$RESUME_EPOCH" != "1500" ]]; then
  echo "Expected last.pt at epoch 1500, found epoch $RESUME_EPOCH" >&2
  exit 1
fi

echo "Resuming $RUN_NAME from epoch $RESUME_EPOCH through epoch 1800"
CUDA_VISIBLE_DEVICES=0,1 "$PYTHON_BIN" train_diffusion.py \
  --dataset_root "$DATASET_ROOT" \
  --output_dir "$RUN_DIR" \
  --epochs 1800 \
  --resume "$RESUME_CHECKPOINT" \
  --batch_size 16 \
  --lr 7.5e-5 \
  --timesteps 800 \
  --ddim_steps 80 \
  --loss_type mse \
  --augmentation_prob 1.0 \
  --seed 42 \
  --device cuda:0 \
  --use_data_parallel true \
  --gpu_ids 0,1 \
  --sample_every 25 \
  --sample_count 4 \
  --save_every 25 \
  --amp true

FINAL_EPOCH="$($PYTHON_BIN -c 'import sys, torch; state = torch.load(sys.argv[1], map_location="cpu", weights_only=False); print(int(state["epoch"]))' "$RUN_DIR/last.pt")"
if [[ "$FINAL_EPOCH" != "1800" ]]; then
  echo "Training did not finish at epoch 1800 (last.pt reports $FINAL_EPOCH)" >&2
  exit 1
fi

IFS=',' read -r -a EPOCHS <<< "$GRID"
for epoch in "${EPOCHS[@]}"; do
  checkpoint="$RUN_DIR/checkpoints/epoch_$(printf '%04d' "$epoch").pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Required evaluation checkpoint is missing: $checkpoint" >&2
    exit 1
  fi
done

echo "Training complete. Evaluating epochs $GRID with 30 samples and cond_batch=4"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" evaluate_matrix.py \
  --runs_root "$RUNS_ROOT" \
  --dataset_root "$DATASET_ROOT" \
  --device cuda:0 \
  --num_samples 30 \
  --grid "$GRID" \
  --only lr_low \
  --grid_only \
  --force \
  --amp \
  --cond_batch 4 \
  --no_summary

echo "Pipeline complete. Training previews: $RUN_DIR/samples"
echo "Evaluation maps: $RUN_DIR/prediction_epoch_<epoch>_samples030"
