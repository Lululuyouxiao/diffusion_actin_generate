#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 7 ]]; then
  echo "Usage: $0 DATASET_ROOT OUTPUT_ROOT [CONDITION_DIR] [MATRIX_JSON] [GPU_IDS] [NUM_SAMPLES] [SEED]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="$(cd "$1" && pwd)"
OUTPUT_ROOT="$(mkdir -p "$2" && cd "$2" && pwd)"
CONDITION_DIR="${3:-}"
MATRIX_JSON="${4:-$PROJECT_DIR/configs/experiment_matrix.json}"
GPU_IDS="${5:-0,1}"
NUM_SAMPLES="${6:-100}"
SEED="${7:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"

if [[ -z "$CONDITION_DIR" ]]; then
  if [[ -d "$DATASET_ROOT/validation/contour" ]]; then
    CONDITION_DIR="$DATASET_ROOT/validation/contour"
  else
    CONDITION_DIR="$DATASET_ROOT/contour"
  fi
fi
CONDITION_DIR="$(cd "$CONDITION_DIR" && pwd)"

"$PYTHON_BIN" - "$PROJECT_DIR" "$DATASET_ROOT" "$OUTPUT_ROOT" "$CONDITION_DIR" "$MATRIX_JSON" "$GPU_IDS" "$NUM_SAMPLES" "$SEED" "$DEVICE" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

project_dir = Path(sys.argv[1])
dataset_root = sys.argv[2]
output_root = Path(sys.argv[3])
condition_dir = sys.argv[4]
matrix_path = Path(sys.argv[5])
gpu_ids = sys.argv[6]
num_samples = sys.argv[7]
seed = sys.argv[8]
device = sys.argv[9]

experiments = json.loads(matrix_path.read_text(encoding="utf-8"))
train_script = project_dir / "train_diffusion.py"
evaluate_script = project_dir / "evaluate_diffusion.py"

for experiment in experiments:
    lr_text = format(float(experiment["lr"]), ".8g")
    aug_text = "0" if float(experiment["augmentation_prob"]) == 0 else "1"
    gpu_label = gpu_ids.replace(",", "-")
    run_name = (
        f"{experiment['name']}__ep{experiment['epochs']}"
        f"_bs{experiment['batch_size']}_lr{lr_text}"
        f"_T{experiment['timesteps']}_S{experiment['ddim_steps']}"
        f"_loss-{experiment['loss_type']}_aug{aug_text}_dp-{gpu_label}"
    )
    run_dir = output_root / run_name
    prediction_dir = run_dir / "prediction"

    command = [
        sys.executable,
        str(train_script),
        "--dataset_root", dataset_root,
        "--output_dir", str(run_dir),
        "--device", device,
        "--use_data_parallel", "true",
        "--gpu_ids", gpu_ids,
        "--seed", seed,
    ]
    for key, value in experiment.items():
        if key == "name":
            continue
        command.extend(["--" + key, str(value).lower() if isinstance(value, bool) else str(value)])

    print("Training:", run_name, flush=True)
    subprocess.run(command, cwd=project_dir, check=True)

    checkpoint = run_dir / "best.pt"
    if not checkpoint.exists():
        checkpoint = run_dir / "last.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint found for {run_name}")

    evaluate_command = [
        sys.executable,
        str(evaluate_script),
        "--checkpoint", str(checkpoint),
        "--condition_dir", condition_dir,
        "--output_dir", str(prediction_dir),
        "--num_samples", num_samples,
        "--device", device,
        "--seed", seed,
    ]
    print("Predicting:", run_name, flush=True)
    subprocess.run(evaluate_command, cwd=project_dir, check=True)
PY
