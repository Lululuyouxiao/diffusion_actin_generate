import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import ConditionImageDataset
from diffusion_core import DiffusionProcess
from train_diffusion import _load_checkpoint, build_model, resolve_device
from training_config import TrainingConfig, parse_bool


def _save_probability_map(probability, path):
    image = np.clip(probability * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(image).save(path)


def _normalize_model_state_dict(state_dict):
    """Load checkpoints saved from either a plain model or DataParallel."""
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def evaluate_checkpoint(
    checkpoint,
    condition_dir,
    output_dir,
    num_samples=100,
    device="auto",
    ddim_steps=None,
    threshold=0.0,
    seed=None,
):
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    run_device = resolve_device(device)
    state = _load_checkpoint(checkpoint, run_device)
    config = TrainingConfig.from_dict(state["config"])
    if ddim_steps is None:
        ddim_steps = config.ddim_steps
    model = build_model(config).to(run_device)
    model.load_state_dict(_normalize_model_state_dict(state["model"]))
    model.eval()
    diffusion = DiffusionProcess(config.timesteps).to(run_device)
    dataset = ConditionImageDataset(condition_dir, resolution=config.resolution)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    base_seed = config.seed if seed is None else int(seed)
    metadata = []

    with torch.no_grad():
        for index, (condition, name) in enumerate(loader):
            condition = condition.to(run_device)
            repeated_condition = condition.repeat(num_samples, 1, 1, 1)
            generated = diffusion.sample_ddim(
                model,
                repeated_condition,
                shape=repeated_condition.shape,
                steps=ddim_steps,
                eta=0.0,
                clip_denoised=True,
                seed=base_seed + index,
            )
            binary_samples = (generated > threshold).float()
            probability = binary_samples.mean(dim=0)[0].cpu().numpy()
            item_dir = output_root / "{:04d}".format(index)
            item_dir.mkdir(parents=True, exist_ok=True)
            np.save(item_dir / "probability_map.npy", probability)
            _save_probability_map(probability, item_dir / "probability_map.png")
            save_image(
                (condition.cpu() + 1.0) / 2.0,
                item_dir / "condition.png",
            )
            save_image(
                (generated[:1].cpu() + 1.0) / 2.0,
                item_dir / "sample_0000.png",
            )
            metadata.append(
                {
                    "index": index,
                    "condition_name": name[0],
                    "num_samples": num_samples,
                    "threshold": threshold,
                    "ddim_steps": ddim_steps,
                }
            )

    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metadata


def main(args=None):
    parser = argparse.ArgumentParser(description="Evaluate repeated conditional diffusion samples")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--condition_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ddim_steps", type=int)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int)
    parsed = parser.parse_args(args)
    evaluate_checkpoint(
        checkpoint=parsed.checkpoint,
        condition_dir=parsed.condition_dir,
        output_dir=parsed.output_dir,
        num_samples=parsed.num_samples,
        device=parsed.device,
        ddim_steps=parsed.ddim_steps,
        threshold=parsed.threshold,
        seed=parsed.seed,
    )


if __name__ == "__main__":
    main()
