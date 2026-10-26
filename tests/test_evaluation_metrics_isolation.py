import tempfile
import unittest
from pathlib import Path

from evaluate_matrix import prepare_metrics_path


class EvaluationMetricsIsolationTests(unittest.TestCase):
    def test_each_sample_count_gets_a_private_copy_of_training_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            original = run_dir / "metrics.csv"
            original.write_text("epoch,train_loss\n2500,0.1\n", encoding="utf-8")

            first = prepare_metrics_path(run_dir, "metrics_epoch_2500_samples025.csv")
            second = prepare_metrics_path(run_dir, "metrics_epoch_2500_samples020.csv")

            self.assertEqual(first.read_text(encoding="utf-8"), original.read_text(encoding="utf-8"))
            self.assertEqual(second.read_text(encoding="utf-8"), original.read_text(encoding="utf-8"))
            first.write_text("changed", encoding="utf-8")
            self.assertNotEqual(second.read_text(encoding="utf-8"), "changed")
            self.assertEqual(original.read_text(encoding="utf-8"),
                             "epoch,train_loss\n2500,0.1\n")

    def test_invalid_metrics_filename_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "metrics.csv").write_text("epoch\n2500\n", encoding="utf-8")
            for filename in ("../other.csv", "other.txt", "metrics_foo/bar.csv"):
                with self.subTest(filename=filename):
                    with self.assertRaises(ValueError):
                        prepare_metrics_path(run_dir, filename)


if __name__ == "__main__":
    unittest.main()
