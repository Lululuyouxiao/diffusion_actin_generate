import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_diffusion import _normalize_model_state_dict, evaluate_checkpoint
from train_diffusion import TrainingConfig, run_training


class EvaluationSmokeTests(unittest.TestCase):
    def test_data_parallel_state_dict_is_normalized_for_single_model(self):
        state = {
            "module.input_block.weight": np.array([1]),
            "output_block.bias": np.array([2]),
        }
        normalized = _normalize_model_state_dict(state)
        self.assertIn("input_block.weight", normalized)
        self.assertIn("output_block.bias", normalized)
        self.assertNotIn("module.input_block.weight", normalized)

    def test_repeated_sampling_writes_probability_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition = root / "condition"
            target = root / "target"
            run_dir = root / "run"
            eval_dir = root / "evaluation"
            condition.mkdir()
            target.mkdir()
            image = np.zeros((16, 16), dtype=np.uint8)
            for index in range(2):
                Image.fromarray(image + index).save(condition / "{:04d}.png".format(index))
                Image.fromarray(255 - image).save(target / "{:04d}.png".format(index))

            config = TrainingConfig(
                condition_dir=str(condition),
                target_dir=str(target),
                output_dir=str(run_dir),
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
                seed=3,
            )
            run_training(config)
            evaluate_checkpoint(
                checkpoint=str(run_dir / "last.pt"),
                condition_dir=str(condition),
                output_dir=str(eval_dir),
                num_samples=2,
                device="cpu",
            )

            self.assertTrue((eval_dir / "0000" / "probability_map.png").exists())
            self.assertTrue((eval_dir / "0000" / "probability_map.npy").exists())
            self.assertTrue((eval_dir / "metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
