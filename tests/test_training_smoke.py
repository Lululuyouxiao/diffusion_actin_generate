import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from train_diffusion import TrainingConfig, build_model, run_training


class TrainingSmokeTests(unittest.TestCase):
    def test_rng_states_are_moved_to_cpu_before_restore(self):
        from train_diffusion import _restore_rng_state

        class MappedState:
            def __init__(self, cpu_state):
                self.cpu_state = cpu_state

            def cpu(self):
                return self.cpu_state

        cpu_state = object()
        cuda_states = [object(), object()]
        checkpoint = {
            "rng_state": MappedState(cpu_state),
            "cuda_rng_state": [MappedState(state) for state in cuda_states],
        }

        with mock.patch("train_diffusion.torch.set_rng_state") as set_cpu_rng, mock.patch(
            "train_diffusion.torch.cuda.is_available", return_value=True
        ), mock.patch("train_diffusion.torch.cuda.set_rng_state_all") as set_cuda_rng:
            _restore_rng_state(checkpoint)

        set_cpu_rng.assert_called_once_with(cpu_state)
        set_cuda_rng.assert_called_once_with(cuda_states)

    def test_multi_gpu_configuration_parses_ids_without_enabling_cpu_parallel(self):
        config = TrainingConfig(device="cpu", gpu_ids="0,1", use_data_parallel=False)
        model = build_model(config)
        wrapped = __import__("train_diffusion").maybe_wrap_data_parallel(model, config)
        self.assertEqual(config.gpu_ids, (0, 1))
        self.assertIs(wrapped, model)

    def test_best_checkpoint_is_not_replaced_by_worse_epoch(self):
        from train_diffusion import _best_checkpoint_update
        self.assertFalse(_best_checkpoint_update(1.0, 2.0)[0])
        self.assertEqual(_best_checkpoint_update(1.0, 2.0)[1], 1.0)
        self.assertTrue(_best_checkpoint_update(2.0, 1.0)[0])
        self.assertEqual(_best_checkpoint_update(2.0, 1.0)[1], 1.0)

    def _write_image(self, path, value):
        array = np.full((16, 16), value, dtype=np.uint8)
        Image.fromarray(array).save(path)

    def test_one_epoch_writes_checkpoint_metrics_and_resume_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition = root / "condition"
            target = root / "target"
            output = root / "run"
            condition.mkdir()
            target.mkdir()
            for index in range(2):
                self._write_image(condition / f"{index:04d}.png", index * 20)
                self._write_image(target / f"{index:04d}.png", 255 - index * 20)

            config = TrainingConfig(
                condition_dir=str(condition),
                target_dir=str(target),
                output_dir=str(output),
                resolution=16,
                batch_size=1,
                epochs=1,
                timesteps=8,
                ddim_steps=4,
                num_workers=0,
                model_dim=8,
                dim_mults=(1, 2),
                amp=False,
                device="cpu",
                sample_every=0,
                seed=7,
            )
            result = run_training(config)

            self.assertEqual(result["epoch"], 1)
            self.assertTrue((output / "last.pt").exists())
            self.assertTrue((output / "metrics.csv").exists())
            with (output / "metrics.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertIn("train_loss", rows[0])

            resumed = config.__class__(**{**config.to_dict(), "epochs": 2, "resume": str(output / "last.pt")})
            resumed_result = run_training(resumed)
            self.assertEqual(resumed_result["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
