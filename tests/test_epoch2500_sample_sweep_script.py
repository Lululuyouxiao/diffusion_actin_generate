import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_DIR / "scripts" / "run_epoch2500_sample_sweep.sh"
RUN_NAME = "lr_low__ep1500_bs16_lr7.5e-05_T800_S80_loss-mse_aug1_dp-0-1"


def bash_path(path):
    path = Path(path).resolve()
    if os.name == "nt":
        return "/{}{}".format(path.drive[0].lower(), path.as_posix()[2:])
    return str(path)


class Epoch2500SampleSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = shutil.which("bash")
        if os.name == "nt" and Path("D:/Git/bin/bash.exe").exists():
            cls.bash = "D:/Git/bin/bash.exe"
        if not cls.bash:
            raise unittest.SkipTest("Bash is unavailable")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT_DIR)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "runs" / RUN_NAME
        self.dataset = self.root / "dataset"
        (self.run / "checkpoints").mkdir(parents=True)
        self.dataset.mkdir()
        (self.run / "checkpoints" / "epoch_2500.pt").touch()
        (self.run / "metrics.csv").write_text("epoch\n2500\n", encoding="ascii")
        self.calls = self.root / "calls.txt"
        self.fake_python = self.root / "fake-python"
        self.fake_python.write_bytes(b"""#!/usr/bin/env bash
set -euo pipefail
printf '%s|%s\\n' "${CUDA_VISIBLE_DEVICES-}" "$*" >> "$CALL_LOG"
if [[ "${FAIL_SAMPLES-}" != "" && "$*" == *"--num_samples $FAIL_SAMPLES "* ]]; then
  exit 9
fi
""")
        self.fake_python.chmod(0o755)

    def run_script(self, **extra_env):
        env = os.environ.copy()
        env.update({"PYTHON_BIN": bash_path(self.fake_python),
                    "CALL_LOG": bash_path(self.calls)})
        env.update(extra_env)
        return subprocess.run(
            [self.bash, bash_path(SCRIPT), bash_path(self.run), bash_path(self.dataset)],
            cwd=PROJECT_DIR, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )

    def logged_calls(self):
        return self.calls.read_text(encoding="utf-8").splitlines() if self.calls.exists() else []

    def test_runs_five_sample_counts_on_epoch2500_validation_set(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = self.logged_calls()
        self.assertEqual(len(calls), 5)
        for call, samples in zip(calls, (25, 20, 15, 10, 5)):
            self.assertTrue(call.startswith("0|evaluate_matrix.py "), call)
            self.assertIn("--grid 2500", call)
            self.assertIn("--num_samples {} ".format(samples), call)
            self.assertIn("--grid_only", call)
            self.assertIn("--force", call)
            self.assertIn("--only " + RUN_NAME, call)
            self.assertIn("--dataset_root " + bash_path(self.dataset), call)
            self.assertNotIn("--limit", call)
            self.assertTrue(
                (self.run / "metrics_epoch_2500_samples{:03d}.csv".format(samples)).exists()
            )

    def test_missing_checkpoint_stops_before_evaluation(self):
        (self.run / "checkpoints" / "epoch_2500.pt").unlink()
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("epoch_2500.pt", result.stderr)
        self.assertEqual(self.logged_calls(), [])

    def test_failure_stops_before_next_sample_count(self):
        result = self.run_script(FAIL_SAMPLES="20")
        self.assertNotEqual(result.returncode, 0)
        calls = self.logged_calls()
        self.assertEqual(len(calls), 2)
        self.assertIn("--num_samples 25 ", calls[0])
        self.assertIn("--num_samples 20 ", calls[1])


if __name__ == "__main__":
    unittest.main()
