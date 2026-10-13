from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    if timesteps <= 1:
        raise ValueError("timesteps must be greater than one")
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=1e-4, max=0.9999)


def extract(values: torch.Tensor, time: torch.Tensor, shape):
    if time.dtype != torch.long:
        time = time.long()
    gathered = values.to(time.device).gather(0, time)
    return gathered.reshape(time.shape[0], *((1,) * (len(shape) - 1)))


def make_ddim_timesteps(timesteps: int, steps: int) -> np.ndarray:
    if steps <= 0 or steps > timesteps:
        raise ValueError("steps must be in [1, timesteps]")
    if timesteps % steps == 0:
        values = np.arange(0, timesteps, timesteps // steps, dtype=np.int64)
    else:
        values = np.rint(np.linspace(0, timesteps - 1, steps)).astype(np.int64)
    values = np.unique(values)
    if len(values) != steps or values[0] != 0:
        raise ValueError("Unable to construct a strictly valid DDIM timestep sequence")
    return values


class DiffusionProcess(torch.nn.Module):
    def __init__(self, timesteps: int = 800):
        super().__init__()
        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.timesteps = int(timesteps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(self, x_start, time, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, time, x_start.shape)
        sqrt_one_minus = extract(self.sqrt_one_minus_alphas_cumprod, time, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def epsilon_loss(self, model, x_start, condition, time, noise=None, loss_type="mse"):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, time=time, noise=noise)
        predicted_noise = model(x_noisy, time, condition)
        if loss_type == "mse":
            return F.mse_loss(predicted_noise, noise)
        if loss_type == "l1":
            return F.l1_loss(predicted_noise, noise)
        if loss_type == "huber":
            return F.smooth_l1_loss(predicted_noise, noise)
        raise ValueError("Unsupported loss_type: {}".format(loss_type))

    @torch.no_grad()
    def sample_ddim(
        self,
        model,
        condition,
        shape: Optional[Sequence[int]] = None,
        steps: int = 80,
        eta: float = 0.0,
        clip_denoised: bool = True,
        seed: Optional[int] = None,
    ):
        if shape is None:
            shape = condition.shape
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = condition.device
        condition = condition.to(device)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
        sample = torch.randn(tuple(shape), device=device, generator=generator)
        timestep_sequence = make_ddim_timesteps(self.timesteps, steps)

        for index in reversed(range(len(timestep_sequence))):
            current_value = int(timestep_sequence[index])
            previous_value = int(timestep_sequence[index - 1]) if index > 0 else -1
            current = torch.full(
                (shape[0],),
                current_value,
                device=device,
                dtype=torch.long,
            )
            alpha_current = extract(self.alphas_cumprod, current, sample.shape)
            if previous_value < 0:
                alpha_previous = torch.ones_like(alpha_current)
            else:
                previous = torch.full(
                    (shape[0],),
                    previous_value,
                    device=device,
                    dtype=torch.long,
                )
                alpha_previous = extract(self.alphas_cumprod, previous, sample.shape)

            predicted_noise = model(sample, current, condition)
            predicted_x0 = (
                sample - torch.sqrt(1.0 - alpha_current) * predicted_noise
            ) / torch.sqrt(alpha_current)
            if clip_denoised:
                predicted_x0 = predicted_x0.clamp(-1.0, 1.0)

            sigma = eta * torch.sqrt(
                ((1.0 - alpha_previous) / (1.0 - alpha_current))
                * (1.0 - alpha_current / alpha_previous)
            )
            direction = torch.sqrt(
                torch.clamp(1.0 - alpha_previous - sigma.square(), min=0.0)
            ) * predicted_noise
            if generator is None:
                random_noise = torch.randn_like(sample)
            else:
                random_noise = torch.randn(
                    sample.shape,
                    device=device,
                    dtype=sample.dtype,
                    generator=generator,
                )
            sample = (
                torch.sqrt(alpha_previous) * predicted_x0
                + direction
                + sigma * random_noise
            )
        return sample

    @torch.no_grad()
    def sample_ddpm(self, model, condition, shape: Optional[Sequence[int]] = None, seed=None):
        if shape is None:
            shape = condition.shape
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = condition.device
        condition = condition.to(device)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
        sample = torch.randn(tuple(shape), device=device, generator=generator)
        for value in reversed(range(self.timesteps)):
            time = torch.full(
                (shape[0],),
                value,
                device=device,
                dtype=torch.long,
            )
            beta_t = extract(self.betas, time, sample.shape)
            sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, time, sample.shape)
            sqrt_one_minus_t = extract(
                self.sqrt_one_minus_alphas_cumprod,
                time,
                sample.shape,
            )
            predicted_noise = model(sample, time, condition)
            mean = sqrt_recip_alpha_t * (
                sample - beta_t * predicted_noise / sqrt_one_minus_t
            )
            if value == 0:
                sample = mean
            else:
                variance = extract(self.posterior_variance, time, sample.shape)
                if generator is None:
                    noise = torch.randn_like(sample)
                else:
                    noise = torch.randn(
                        sample.shape,
                        device=device,
                        dtype=sample.dtype,
                        generator=generator,
                    )
                sample = mean + torch.sqrt(variance) * noise
        return sample
