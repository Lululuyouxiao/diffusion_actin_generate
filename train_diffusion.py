import csv
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets import PairedImageDataset, discover_split_directories, pair_image_files, split_pairs
from diffusion_core import DiffusionProcess
from network import Unet
from training_config import TrainingConfig, config_from_args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available")
    return device


def _resolve_training_pairs(config):
    if config.condition_dir and config.target_dir:
        train_pairs = pair_image_files(config.condition_dir, config.target_dir)
    elif config.dataset_root:
        condition_dir, target_dir = discover_split_directories(config.dataset_root, "train")
        train_pairs = pair_image_files(condition_dir, target_dir)
    else:
        raise ValueError("Provide either dataset_root or both condition_dir and target_dir")

    if config.val_condition_dir and config.val_target_dir:
        validation_pairs = pair_image_files(config.val_condition_dir, config.val_target_dir)
    elif config.dataset_root:
        root = Path(config.dataset_root)
        validation_root = root / "validation"
        if validation_root.is_dir():
            condition_dir, target_dir = discover_split_directories(root, "validation")
            validation_pairs = pair_image_files(condition_dir, target_dir)
        else:
            train_pairs, validation_pairs = split_pairs(
                train_pairs,
                config.val_fraction,
                config.seed,
            )
    else:
        train_pairs, validation_pairs = split_pairs(
            train_pairs,
            config.val_fraction,
            config.seed,
        )
    if not train_pairs or not validation_pairs:
        raise ValueError("Both training and validation datasets must contain at least one pair")
    return train_pairs, validation_pairs


def build_dataloaders(config):
    train_pairs, validation_pairs = _resolve_training_pairs(config)
    train_dataset = PairedImageDataset(
        train_pairs,
        resolution=config.resolution,
        augment=True,
        augmentation_prob=config.augmentation_prob,
    )
    validation_dataset = PairedImageDataset(
        validation_pairs,
        resolution=config.resolution,
        augment=False,
        augmentation_prob=0.0,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    common = {
        "num_workers": config.num_workers,
        "pin_memory": config.device != "cpu" and torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, validation_loader


def build_model(config):
    return Unet(
        dim=config.model_dim,
        channels=1,
        dim_mults=config.dim_mults,
        resnet_block_groups=config.groups,
    )


def maybe_wrap_data_parallel(model, config):
    if not config.use_data_parallel:
        return model
    if not torch.cuda.is_available():
        raise RuntimeError("DataParallel training requires CUDA")
    available = torch.cuda.device_count()
    if max(config.gpu_ids) >= available:
        raise RuntimeError(
            "Requested gpu_ids {} but only {} CUDA device(s) are available".format(
                config.gpu_ids, available
            )
        )
    return torch.nn.DataParallel(model, device_ids=list(config.gpu_ids))


def _best_checkpoint_update(best_val_loss, val_loss):
    improved = val_loss < best_val_loss
    return improved, min(best_val_loss, val_loss)


def _autocast(device, enabled):
    if device.type == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _checkpoint_state(
    config,
    model,
    optimizer,
    scaler,
    epoch,
    global_step,
    best_val_loss,
):
    state = {
        "config": config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(checkpoint):
    if "rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng_state"]]
        )


def _validate_resume_config(current, saved):
    critical_fields = (
        "resolution",
        "timesteps",
        "model_dim",
        "dim_mults",
        "groups",
        "loss_type",
    )
    mismatches = []
    for field in critical_fields:
        current_value = getattr(current, field)
        saved_value = saved.get(field)
        if field == "dim_mults":
            current_value = list(current_value)
            saved_value = list(saved_value or [])
        if current_value != saved_value:
            mismatches.append(
                "{}: requested={!r}, checkpoint={!r}".format(
                    field,
                    current_value,
                    saved_value,
                )
            )
    if mismatches:
        raise ValueError(
            "Resume checkpoint is incompatible with the requested configuration: "
            + "; ".join(mismatches)
        )


def _write_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def _make_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _run_epoch(model, diffusion, loader, optimizer, scaler, config, device, training):
    model.train(training)
    total_loss = 0.0
    total_items = 0
    for batch_index, (condition, target) in enumerate(loader):
        if config.max_train_batches and training and batch_index >= config.max_train_batches:
            break
        if config.max_val_batches and not training and batch_index >= config.max_val_batches:
            break
        condition = condition.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        time = torch.randint(
            0,
            diffusion.timesteps,
            (target.shape[0],),
            device=device,
            dtype=torch.long,
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        with _autocast(device, scaler.is_enabled()):
            loss = diffusion.epsilon_loss(
                model,
                target,
                condition,
                time,
                loss_type=config.loss_type,
            )
        if training:
            scaler.scale(loss).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        count = target.shape[0]
        total_loss += float(loss.detach().cpu()) * count
        total_items += count
    if total_items == 0:
        raise RuntimeError("No batches were processed")
    return total_loss / total_items


def _save_preview(model, diffusion, loader, output_dir, config, device, epoch):
    condition, target = next(iter(loader))
    condition = condition[: config.sample_count].to(device)
    target = target[: config.sample_count].to(device)
    model.eval()
    generated = diffusion.sample_ddim(
        model,
        condition,
        shape=target.shape,
        steps=config.ddim_steps,
        eta=0.0,
        seed=config.seed + epoch,
    )
    preview_dir = Path(output_dir) / "samples"
    preview_dir.mkdir(parents=True, exist_ok=True)
    save_image((condition.cpu() + 1.0) / 2.0, preview_dir / "epoch_{:04d}_condition.png".format(epoch))
    save_image((target.cpu() + 1.0) / 2.0, preview_dir / "epoch_{:04d}_target.png".format(epoch))
    save_image((generated.cpu() + 1.0) / 2.0, preview_dir / "epoch_{:04d}_generated.png".format(epoch))


def run_training(config: TrainingConfig):
    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(output_dir / "config.json")

    train_loader, validation_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    model = maybe_wrap_data_parallel(model, config)
    diffusion = DiffusionProcess(config.timesteps).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = _make_scaler(amp_enabled)

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    if config.resume:
        checkpoint = _load_checkpoint(config.resume, device)
        _validate_resume_config(
            config,
            checkpoint.get("config", {}),
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        _restore_rng_state(checkpoint)

    metrics_path = output_dir / "metrics.csv"
    write_header = not metrics_path.exists() or start_epoch == 1
    metrics_file = metrics_path.open("a", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(
        metrics_file,
        fieldnames=("epoch", "global_step", "train_loss", "val_loss", "learning_rate"),
    )
    if write_header:
        metrics_writer.writeheader()

    try:
        for epoch in range(start_epoch, config.epochs + 1):
            train_loss = _run_epoch(
                model,
                diffusion,
                train_loader,
                optimizer,
                scaler,
                config,
                device,
                training=True,
            )
            with torch.no_grad():
                val_loss = _run_epoch(
                    model,
                    diffusion,
                    validation_loader,
                    optimizer,
                    scaler,
                    config,
                    device,
                    training=False,
                )
            global_step += len(train_loader)
            metrics_writer.writerow(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": "{:.8f}".format(train_loss),
                    "val_loss": "{:.8f}".format(val_loss),
                    "learning_rate": "{:.10g}".format(optimizer.param_groups[0]["lr"]),
                }
            )
            metrics_file.flush()

            improved, best_val_loss = _best_checkpoint_update(best_val_loss, val_loss)
            state = _checkpoint_state(
                config,
                model,
                optimizer,
                scaler,
                epoch,
                global_step,
                best_val_loss,
            )
            _write_checkpoint(output_dir / "last.pt", state)
            if improved:
                _write_checkpoint(output_dir / "best.pt", state)
            if config.save_every and epoch % config.save_every == 0:
                _write_checkpoint(output_dir / "checkpoints" / "epoch_{:04d}.pt".format(epoch), state)
            if config.sample_every and epoch % config.sample_every == 0:
                _save_preview(
                    model,
                    diffusion,
                    validation_loader,
                    output_dir,
                    config,
                    device,
                    epoch,
                )
    finally:
        metrics_file.close()

    return {
        "epoch": config.epochs,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "output_dir": str(output_dir),
    }


def main(args=None):
    config = config_from_args(args)
    result = run_training(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
