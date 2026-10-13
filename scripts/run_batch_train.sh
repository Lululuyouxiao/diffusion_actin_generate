#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 DATASET_ROOT OUTPUT_ROOT [MATRIX_JSON]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="$(cd "$1" && pwd)"
OUTPUT_ROOT="$(mkdir -p "$2" && cd "$2" && pwd)"
MATRIX_JSON="${3:-$PROJECT_DIR/configs/experiment_matrix.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"

"$PYTHON_BIN" - "$PROJECT_DIR" "$DATASET_ROOT" "$OUTPUT_ROOT" "$MATRIX_JSON" "$DEVICE" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

project_dir = Path(sys.argv[1])
dataset_root = sys.argv[2]
output_root = Path(sys.argv[3])
matrix_path = Path(sys.argv[4])
device = sys.argv[5]

experiments = json.loads(matrix_path.read_text(encoding="utf-8"))
train_script = project_dir / "train_diffusion.py"

for experiment in experiments:
    name = experiment["name"]
    run_dir = output_root / name
    command = [
        sys.executable,
        str(train_script),
        "--dataset_root",
        dataset_root,
        "--output_dir",
        str(run_dir),
        "--device",
        device,
    ]
    for key, value in experiment.items():
        if key == "name":
            continue
        command.append("--" + key)
        if isinstance(value, bool):
            command.append("true" if value else "false")
        elif isinstance(value, list):
            command.append(",".join(str(item) for item in value))
        else:
            command.append(str(value))
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=project_dir, check=True)
PY
