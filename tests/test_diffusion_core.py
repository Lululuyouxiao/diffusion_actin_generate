import unittest

import torch
from torch import nn

from diffusion_core import DiffusionProcess, make_ddim_timesteps


class ZeroNoiseModel(nn.Module):
    def forward(self, x, time, condition):
        return torch.zeros_like(x)


class DiffusionCoreTests(unittest.TestCase):
    def test_cosine_schedule_and_zero_based_timesteps(self):
        process = DiffusionProcess(timesteps=800)
        self.assertEqual(tuple(process.betas.shape), (800,))
        self.assertGreater(float(process.betas.min()), 0.0)
        self.assertLess(float(process.betas.max()), 1.0)

        timesteps = make_ddim_timesteps(800, 80)
        self.assertEqual(len(timesteps), 80)
        self.assertEqual(timesteps[0], 0)
        self.assertEqual(timesteps[-1], 790)
        self.assertTrue(all(earlier < later for earlier, later in zip(timesteps, timesteps[1:])))

    def test_q_sample_and_loss_preserve_shapes(self):
        process = DiffusionProcess(timesteps=20)
        x_start = torch.zeros(2, 1, 8, 8)
        condition = torch.zeros_like(x_start)
        time = torch.tensor([0, 19], dtype=torch.long)
        noise = torch.ones_like(x_start)

        noisy = process.q_sample(x_start, time, noise=noise)
        self.assertEqual(noisy.shape, x_start.shape)
        loss = process.epsilon_loss(ZeroNoiseModel(), x_start, condition, time, noise=noise)
        self.assertAlmostEqual(float(loss), 1.0, places=5)

    def test_ddim_eta_zero_is_deterministic(self):
        process = DiffusionProcess(timesteps=20)
        model = ZeroNoiseModel()
        condition = torch.zeros(1, 1, 8, 8)
        first = process.sample_ddim(model, condition, steps=5, eta=0.0, seed=123)
        second = process.sample_ddim(model, condition, steps=5, eta=0.0, seed=123)
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
