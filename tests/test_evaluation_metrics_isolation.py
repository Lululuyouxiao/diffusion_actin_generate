import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from evaluate_matrix import (
    load_eval_data,
    prepare_metrics_path,
    maybe_wrap_data_parallel,
)
from training_config import TrainingConfig


class EvaluationMetricsIsolationTests(unittest.TestCase):
    def test_evaluation_can_wrap_model_across_requested_gpus(self):
        model = torch.nn.Identity()
        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.cuda.device_count", return_value=2), \
             mock.patch("torch.nn.DataParallel", return_value="wrapped") as data_parallel:
            wrapped = maybe_wrap_data_parallel(model, torch.device("cuda:0"), (0, 1))

        self.assertEqual(wrapped, "wrapped")
        data_parallel.assert_called_once_with(model, device_ids=[0, 1])

    def test_all_data_mode_keeps_every_paired_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition = root / "contour"
            target = root / "SF"
            condition.mkdir()
            target.mkdir()
            for index in range(10):
                image = Image.fromarray(np.zeros((16, 16), dtype=np.uint8))
                image.save(condition / "{:04d}.png".format(index))
                image.save(target / "{:04d}.png".format(index))
            config = TrainingConfig(resolution=16, model_dim=8, dim_mults=(1, 2), groups=8)

            all_data = load_eval_data(root, config, all_data=True)
            validation_data = load_eval_data(root, config, all_data=False)

            self.assertEqual(len(all_data), 10)
            self.assertEqual(len(validation_data), 1)

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
