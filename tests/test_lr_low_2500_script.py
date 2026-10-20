import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_DIR / "scripts" / "run_lr_low_2500_then_eval.sh"
RUN_NAME = "lr_low__ep1500_bs16_lr7.5e-05_T800_S80_loss-mse_aug1_dp-0-1"
EPOCHS = list(range(1850, 2501, 50))


def bash_path(path):
    path = Path(path).resolve()
    if os.name == "nt":
        return "/{}{}".format(path.drive[0].lower(), path.as_posix()[2:])
    return str(path)


class LrLow2500ScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = shutil.which("bash")
        if os.name == "nt":
            candidate = Path("D:/Git/bin/bash.exe")
            if candidate.exists():
                cls.bash = str(candidate)
        if not cls.bash:
            raise unittest.SkipTest("Bash is unavailable")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT_DIR)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runs = self.root / "runs"
        self.dataset = self.root / "dataset"
        self.run = self.runs / RUN_NAME
        self.run.mkdir(parents=True)
        self.dataset.mkdir()
        (self.run / "last.pt").write_text("1800", encoding="ascii")
        (self.run / "metrics.csv").write_text("epoch,train_loss\n1800,0.1\n", encoding="ascii")
        self.call_log = self.root / "calls.txt"
        self.fake_python = self.root / "fake-python"
        self.fake_python.write_bytes(b"""#!/usr/bin/env bash
set -euo pipefail
printf '%s|%s\\n' "${CUDA_VISIBLE_DEVICES-}" "$*" >> "$CALL_LOG"
case "$1" in
  -c)
    cat "$FAKE_EPOCH_FILE"
    ;;
  train_diffusion.py)
    echo 2500 > "$FAKE_EPOCH_FILE"
    mkdir -p "$FAKE_RUN_DIR/checkpoints"
    for epoch in $(seq 1850 50 2500); do
      if [[ "$epoch" != "${FAKE_MISSING_EPOCH-}" ]]; then
        touch "$FAKE_RUN_DIR/checkpoints/epoch_$(printf '%04d' "$epoch").pt"
      fi
    done
    ;;
  evaluate_matrix.py)
    ;;
  *)
    echo "Unexpected command: $1" >&2
    exit 3
    ;;
esac
""")
        self.fake_python.chmod(0o755)

    def run_script(self, **extra_env):
        env = os.environ.copy()
        env.update({
            "PYTHON_BIN": bash_path(self.fake_python),
            "CALL_LOG": bash_path(self.call_log),
            "FAKE_EPOCH_FILE": bash_path(self.run / "last.pt"),
            "FAKE_RUN_DIR": bash_path(self.run),
        })
        env.update(extra_env)
        return subprocess.run(
            [self.bash, bash_path(SCRIPT), bash_path(self.runs), bash_path(self.dataset)],
            cwd=PROJECT_DIR,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    def calls(self):
        if not self.call_log.exists():
            return []
        return self.call_log.read_text(encoding="utf-8").splitlines()

    def test_resume_then_evaluate_only_requested_grid(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.calls()
        training = [line for line in calls if "|train_diffusion.py " in line]
        evaluation = [line for line in calls if "|evaluate_matrix.py " in line]
        self.assertEqual(len(training), 1)
        self.assertEqual(len(evaluation), 1)
        self.assertLess(calls.index(training[0]), calls.index(evaluation[0]))
        self.assertTrue(training[0].startswith("0,1|"))
        self.assertIn("--epochs 2500", training[0])
        self.assertIn("--resume ", training[0])
        self.assertIn("--save_every 25", training[0])
        self.assertIn("--sample_every 25", training[0])
        self.assertTrue(evaluation[0].startswith("0|"))
        self.assertIn("--grid " + ",".join(map(str, EPOCHS)), evaluation[0])
        self.assertIn("--num_samples 30", evaluation[0])
        self.assertIn("--cond_batch 4", evaluation[0])
        self.assertIn("--grid_only", evaluation[0])
        self.assertIn("--only " + RUN_NAME, evaluation[0])
        self.assertNotIn("--force", evaluation[0])
        self.assertNotIn("--limit", evaluation[0])

    def test_missing_checkpoint_stops_before_evaluation(self):
        result = self.run_script(FAKE_MISSING_EPOCH="2300")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("epoch_2300.pt", result.stderr)
        self.assertFalse(any("|evaluate_matrix.py " in line for line in self.calls()))

    def test_completed_training_can_restart_evaluation(self):
        (self.run / "last.pt").write_text("2500", encoding="ascii")
        checkpoints = self.run / "checkpoints"
        checkpoints.mkdir()
        for epoch in EPOCHS:
            (checkpoints / "epoch_{:04d}.pt".format(epoch)).touch()
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(any("|train_diffusion.py " in line for line in self.calls()))
        self.assertTrue(any("|evaluate_matrix.py " in line for line in self.calls()))

    def test_earlier_checkpoint_is_rejected(self):
        (self.run / "last.pt").write_text("1500", encoding="ascii")
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("between epoch 1800 and 2500", result.stderr)
        self.assertFalse(any("|train_diffusion.py " in line for line in self.calls()))


if __name__ == "__main__":
    unittest.main()
