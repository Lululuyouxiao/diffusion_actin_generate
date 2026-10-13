from network import *
# from loss_function import *
from tqdm import tqdm
import config
import argparse
import numpy as np

parser = argparse.ArgumentParser()

params = config.parse_args()


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


timesteps = params.timesteps
betas = cosine_beta_schedule(timesteps=timesteps)

alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


@torch.no_grad()
def p_sample(model, x, c, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t, c) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, c, shape):
    device = next(model.parameters()).device

    b = shape[0]
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, params.timesteps)), desc='sampling loop time step', total=params.timesteps):
        img = p_sample(model, img, c, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model, c, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, c, shape=(batch_size, channels, image_size, image_size))


@torch.no_grad()
def ddim_sample(
        model,
        condition,
        image_size,
        batch_size=16,
        channels=3,
        ddim_timesteps=80,
        ddim_discr_method="uniform",
        ddim_eta=0.0,
        clip_denoised=True):
    if ddim_discr_method == 'uniform':
        c = params.timesteps // ddim_timesteps
        ddim_timestep_seq = np.asarray(list(range(0, params.timesteps,c)))
    elif ddim_discr_method == 'quad':
        ddim_timestep_seq = (
                (np.linspace(0, np.sqrt(params.timesteps * .8), ddim_timesteps)) ** 2
        ).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
    ddim_timestep_seq = ddim_timestep_seq + 1
    ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

    device = next(model.parameters()).device
    sample_img = torch.randn((batch_size, channels, image_size, image_size), device=device)
    for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
        t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
        prev_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)

        alpha_cumprod_t = extract(alphas_cumprod, t, sample_img.shape)
        alpha_cumprod_t_prev = extract(alphas_cumprod, prev_t, sample_img.shape)
        
        pred_noise = model(sample_img, t, condition)
        pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
        sigmas_t = ddim_eta * torch.sqrt(
            (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

        pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise

        x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)

        sample_img = x_prev

    return sample_img.cpu().numpy()
