# -*- coding: utf-8 -*-
"""Matrix checkpoint screening evaluation.

For every training-run directory under --runs_root (e.g. one matrix output root
containing paper_baseline__ep1500_..., timesteps_400__..., etc.), evaluate a
fixed grid of checkpoints on the validation split with image-quality metrics,
merge the results into the run's existing metrics.csv (incremental, resumable),
keep the 5+1+2 prediction image sets, and render trend plots.

Metrics (all computed against the paired SF ground truth, validation split only):
    ssim / psnr          -- fidelity of the 30-sample average probability map
    dice / iou           -- binarized at threshold 0.5 (0.3/0.7 kept as dice03/07)
    orient               -- structure-tensor fiber orientation consistency
    outside_mean/frac    -- noise outside the cell silhouette (lower is better)
    score_avg            -- mean of min-max normalized ssim/psnr/dice/iou/orient

Checkpoint grid per run: 250,500,750,1000,1100,1200,1300,1400,1500 + best.pt.

Usage (on the server):
    python evaluate_matrix.py \
        --runs_root  /data/yuhanzhao/diffusion_model/runs/matrix_dp2 \
        --dataset_root /data/yuhanzhao/diffusion_model/Actin-stress-fibers-generator-main/dataset \
        --device cuda:0 --num_samples 30

Useful extras:
    --only paper_baseline            evaluate a single run (smoke test)
    --skip lr_high                   additional runs to skip (default)
    --grid 250,500,...               override the checkpoint grid
    --force                          re-evaluate epochs that already have scores
    --amp --cond_batch 2             faster sampling (mixed precision + packed batch)
    --limit 64                       evaluate only the first N validation pairs
"""

import argparse
import shutil
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

from datasets import (
    _to_tensor,
    discover_split_directories,
    pair_image_files,
    split_pairs,
)
from diffusion_core import DiffusionProcess
from train_diffusion import _load_checkpoint, build_model, resolve_device
from training_config import TrainingConfig

METRIC_COLUMNS = [
    "ssim",
    "psnr",
    "dice",
    "iou",
    "orient",
    "outside_mean",
    "outside_frac",
    "dice03",
    "dice07",
]
SCORE_COLUMNS = ["ssim", "psnr", "dice", "iou", "orient"]
DEFAULT_GRID = [250, 500, 750, 1000, 1100, 1200, 1300, 1400, 1500]
KEEP_ROLES_METRIC = ["ssim", "psnr", "dice", "iou", "orient"]
KEEP_ROLES_SPECIAL = [
    ("avg", "prediction_metricbest_avg"),
    ("bestpt", "prediction_checkpoint_bestpt"),
    ("lastpt", "prediction_checkpoint_lastpt"),
]


def _checkpoint_jobs(run_dir, grid, best_epoch=None, include_best=True):
    """Return checkpoint files for the requested grid.

    ``include_best=False`` is used for targeted evaluations where evaluating
    the grid is the whole job.  In that mode ``best.pt`` is never appended,
    even when its epoch is outside the requested grid.
    """
    run_dir = Path(run_dir)
    jobs = []
    seen = set()
    for epoch in grid:
        epoch = int(epoch)
        path = run_dir / "checkpoints" / "epoch_{:04d}.pt".format(epoch)
        if path.exists() and epoch not in seen:
            jobs.append((epoch, path))
            seen.add(epoch)
    if include_best and best_epoch is not None:
        best_epoch = int(best_epoch)
        best_path = run_dir / "best.pt"
        if best_path.exists() and best_epoch not in seen:
            jobs.append((best_epoch, best_path))
    return jobs
MIN_COHERENT_PIXELS = 50


# --------------------------------------------------------------------------- #
# image metrics (numpy / scipy)
# --------------------------------------------------------------------------- #
def _ssim(a, b):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    sigma = 1.5
    mu_a = gaussian_filter(a, sigma)
    mu_b = gaussian_filter(b, sigma)
    va = np.clip(gaussian_filter(a * a, sigma) - mu_a ** 2, 0.0, None)
    vb = np.clip(gaussian_filter(b * b, sigma) - mu_b ** 2, 0.0, None)
    vab = gaussian_filter(a * b, sigma) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * vab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    return float((num / den).mean())


def _psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _dice_iou(pred_mask, gt_mask):
    inter = float(np.logical_and(pred_mask, gt_mask).sum())
    p_sum = float(pred_mask.sum())
    g_sum = float(gt_mask.sum())
    dice = 2.0 * inter / (p_sum + g_sum + 1e-12)
    iou = inter / (p_sum + g_sum - inter + 1e-12)
    return float(dice), float(iou)


def _structure_tensor(img):
    sigma_d, sigma_w = 1.0, 2.0
    gx = gaussian_filter(img, sigma_d, order=(0, 1))
    gy = gaussian_filter(img, sigma_d, order=(1, 0))
    jxx = gaussian_filter(gx * gx, sigma_w)
    jxy = gaussian_filter(gx * gy, sigma_w)
    jyy = gaussian_filter(gy * gy, sigma_w)
    tmp = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    l1 = 0.5 * (jxx + jyy + tmp)
    l2 = 0.5 * (jxx + jyy - tmp)
    coherence = (l1 - l2) / (l1 + l2 + 1e-12)
    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    return theta, coherence


def _orient_consistency(prob, gt_gray):
    t1, c1 = _structure_tensor(prob)
    t2, c2 = _structure_tensor(gt_gray)
    mask = (c1 > 0.5) & (c2 > 0.5)
    if int(mask.sum()) < MIN_COHERENT_PIXELS:
        return float("nan")
    return float(np.cos(2.0 * (t1 - t2))[mask].mean())


def compute_metrics(prob, condition_np, gt_gray, gt_bin):
    """prob: [0,1] float probability map; condition_np / gt_gray: [-1,1]."""
    cell_mask = condition_np > 0.0
    outside = ~cell_mask
    pred_bin = prob > 0.5
    dice, iou = _dice_iou(pred_bin, gt_bin)
    dice03 = _dice_iou(prob > 0.3, gt_bin)[0]
    dice07 = _dice_iou(prob > 0.7, gt_bin)[0]
    return {
        "ssim": _ssim(prob, (gt_gray + 1.0) / 2.0),
        "psnr": _psnr(prob, (gt_gray + 1.0) / 2.0),
        "dice": dice,
        "iou": iou,
        "orient": _orient_consistency(prob, (gt_gray + 1.0) / 2.0),
        "outside_mean": float(prob[outside].mean()) if outside.any() else 0.0,
        "outside_frac": float((prob[outside] > 0.5).mean()) if outside.any() else 0.0,
        "dice03": float(dice03),
        "dice07": float(dice07),
    }


# --------------------------------------------------------------------------- #
# data / checkpoint helpers
# --------------------------------------------------------------------------- #
def load_val_data(dataset_root, config):
    condition_dir, target_dir = discover_split_directories(dataset_root, "train")
    pairs = pair_image_files(condition_dir, target_dir)
    _, val_pairs = split_pairs(pairs, config.val_fraction, config.seed)
    val_data = []
    for pair in val_pairs:
        condition = _to_tensor(Image.open(pair.condition), config.resolution).numpy()
        target = _to_tensor(Image.open(pair.target), config.resolution).numpy()
        val_data.append(
            {
                "name": Path(pair.condition).name,
                "condition": condition[0],          # [H, W] in [-1, 1]
                "gt_gray": target[0],               # [H, W] in [-1, 1]
                "gt_bin": (target[0] > 0.0),        # fiber = bright
            }
        )
    print("  validation pairs: {}".format(len(val_data)), flush=True)
    return val_data


def _strip_module(state_dict):
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint(path, device):
    state = _load_checkpoint(str(path), device)
    config = TrainingConfig.from_dict(state["config"])
    model = build_model(config).to(device)
    model.load_state_dict(_strip_module(state["model"]))
    model.eval()
    epoch = int(state.get("epoch", 0))
    return model, config, epoch


@torch.no_grad()
def evaluate_checkpoint(model, config, val_data, num_samples, device, seed,
                        amp=False, cond_batch=1):
    diffusion = DiffusionProcess(config.timesteps).to(device)
    use_amp = bool(amp) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    prob_maps = np.zeros((len(val_data), config.resolution, config.resolution), np.float16)
    sample0 = np.zeros((len(val_data), config.resolution, config.resolution), np.uint8)
    metric_sum = {key: 0.0 for key in METRIC_COLUMNS}
    nan_counts = {key: 0 for key in METRIC_COLUMNS}
    count = 0
    for index in range(0, len(val_data), max(1, cond_batch)):
        chunk = val_data[index:index + max(1, cond_batch)]
        repeated = torch.cat(
            [
                torch.from_numpy(item["condition"]).unsqueeze(0).repeat(num_samples, 1, 1, 1)
                for item in chunk
            ],
            dim=0,
        ).to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            generated = diffusion.sample_ddim(
                model,
                repeated,
                shape=repeated.shape,
                steps=config.ddim_steps,
                eta=0.0,
                clip_denoised=True,
                seed=seed + index,
            )
        generated = generated.float()
        for offset, item in enumerate(chunk):
            g = generated[offset * num_samples:(offset + 1) * num_samples]
            prob = (g > 0.0).float().mean(dim=0)[0].cpu().numpy()
            flat_index = index + offset
            prob_maps[flat_index] = prob.astype(np.float16)
            sample0[flat_index] = (
                np.clip((g[0].cpu().numpy() + 1.0) / 2.0 * 255.0, 0, 255)
            ).astype(np.uint8)
            metrics = compute_metrics(prob, item["condition"], item["gt_gray"], item["gt_bin"])
            for key in METRIC_COLUMNS:
                value = metrics[key]
                if np.isnan(value):
                    nan_counts[key] += 1
                    continue
                metric_sum[key] += value
            count += 1
    means = {}
    for key in METRIC_COLUMNS:
        valid = count - nan_counts[key]
        means[key] = metric_sum[key] / valid if valid > 0 else float("nan")
    return prob_maps, sample0, means


# --------------------------------------------------------------------------- #
# metrics.csv incremental merge
# --------------------------------------------------------------------------- #
def read_metrics_csv(path):
    if not path.exists():
        return [], ["epoch"]
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or ["epoch"])


def write_metrics_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: int(float(r.get("epoch", 0) or 0))):
            writer.writerow({col: row.get(col, "") for col in columns})


def merge_epoch_metrics(metrics_path, epoch, metrics):
    rows, columns = read_metrics_csv(metrics_path)
    for col in METRIC_COLUMNS + ["score_avg"]:
        if col not in columns:
            columns.append(col)
    target = None
    for row in rows:
        if int(float(row.get("epoch", -1) or -1)) == epoch:
            target = row
            break
    if target is None:
        target = {"epoch": str(epoch)}
        rows.append(target)
    for key, value in metrics.items():
        if isinstance(value, float) and np.isnan(value):
            target[key] = ""
        else:
            target[key] = "{:.6f}".format(value)
    write_metrics_csv(metrics_path, rows, columns)
    return target


def fill_score_avg(metrics_path):
    rows, columns = read_metrics_csv(metrics_path)
    if "score_avg" not in columns:
        columns.append("score_avg")
    score_rows = []
    for row in rows:
        values = []
        ok = True
        for col in SCORE_COLUMNS:
            raw = row.get(col, "")
            if raw in ("", None):
                ok = False
                break
            values.append(float(raw))
        if ok:
            score_rows.append((row, values))
    if not score_rows:
        return
    norm_stack = []
    for idx in range(len(SCORE_COLUMNS)):
        col_vals = [vals[idx] for _, vals in score_rows]
        vmin, vmax = min(col_vals), max(col_vals)
        if vmax - vmin < 1e-12:
            norm_stack.append([0.5] * len(col_vals))
        else:
            norm_stack.append([(v - vmin) / (vmax - vmin) for v in col_vals])
    for pos, (row, _) in enumerate(score_rows):
        score = float(np.mean([norm_stack[i][pos] for i in range(len(SCORE_COLUMNS))]))
        row["score_avg"] = "{:.6f}".format(score)
    write_metrics_csv(metrics_path, rows, columns)


# --------------------------------------------------------------------------- #
# keep-set saving (5 + 1 + 2)
# --------------------------------------------------------------------------- #
def save_keep_set(run_dir, folder_name, epoch, prob_maps, sample0, val_data,
                  num_samples, ddim_steps):
    out_dir = Path(run_dir) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for index, item in enumerate(val_data):
        item_dir = out_dir / "{:04d}".format(index)
        item_dir.mkdir(parents=True, exist_ok=True)
        np.save(item_dir / "probability_map.npy", prob_maps[index].astype(np.float32))
        Image.fromarray(
            np.clip(prob_maps[index].astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
        ).save(item_dir / "probability_map.png")
        condition_u8 = ((item["condition"] + 1.0) / 2.0 * 255.0).astype(np.uint8)
        Image.fromarray(condition_u8).save(item_dir / "condition.png")
        Image.fromarray(sample0[index]).save(item_dir / "sample_0000.png")
        metadata.append(
            {
                "index": index,
                "condition_name": item["name"],
                "epoch": epoch,
                "num_samples": num_samples,
                "ddim_steps": ddim_steps,
                "threshold": 0.5,
            }
        )
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_run_trend(run_dir, metrics_path, best_epoch, run_name):
    rows, columns = read_metrics_csv(metrics_path)
    if not rows:
        return
    epochs, train_loss, val_loss = [], [], []
    metric_series = {col: [] for col in ["ssim", "dice", "iou", "orient"]}
    for row in rows:
        try:
            ep = int(float(row.get("epoch", 0)))
        except ValueError:
            continue
        epochs.append(ep)
        for name, bucket in (("train_loss", train_loss), ("val_loss", val_loss)):
            raw = row.get(name, "")
            bucket.append(float(raw) if raw not in ("", None) else float("nan"))
        for col in metric_series:
            raw = row.get(col, "")
            metric_series[col].append(float(raw) if raw not in ("", None) else float("nan"))

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    panels = [
        ("train_loss", train_loss),
        ("val_loss", val_loss),
        ("ssim", metric_series["ssim"]),
        ("dice", metric_series["dice"]),
        ("iou", metric_series["iou"]),
        ("orient", metric_series["orient"]),
    ]
    for ax, (title, series) in zip(axes.ravel(), panels):
        ax.plot(epochs, series, marker="o", ms=3, lw=1)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if best_epoch:
            ax.axvline(best_epoch, color="r", ls="--", lw=1, alpha=0.7)
    fig.suptitle("{} (red dashed = best.pt epoch {})".format(run_name, best_epoch))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(run_dir) / "evaluation_trend.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_matrix_summary(runs_root, run_results):
    """run_results: list of (run_name, metrics_path)."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    panels = [("dice", "Dice"), ("iou", "IoU"), ("ssim", "SSIM"), ("orient", "Orient consistency")]
    for ax, (col, title) in zip(axes.ravel(), panels):
        for run_name, metrics_path in run_results:
            rows, _ = read_metrics_csv(metrics_path)
            xs, ys = [], []
            for row in rows:
                raw = row.get(col, "")
                if raw in ("", None):
                    continue
                xs.append(int(float(row.get("epoch", 0))))
                ys.append(float(raw))
            if xs:
                ax.plot(xs, ys, marker="o", ms=3, lw=1, label=run_name)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Matrix screening summary")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(runs_root) / "evaluation_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# per-run driver
# --------------------------------------------------------------------------- #
def prepare_metrics_path(run_dir, metrics_filename):
    run_dir = Path(run_dir)
    source = run_dir / "metrics.csv"
    if not source.exists():
        raise FileNotFoundError("Missing training metrics: {}".format(source))
    if metrics_filename == "metrics.csv":
        return source
    if (Path(metrics_filename).name != metrics_filename
            or not metrics_filename.startswith("metrics_")
            or not metrics_filename.endswith(".csv")):
        raise ValueError("metrics_filename must be a metrics_*.csv basename")
    target = run_dir / metrics_filename
    if not target.exists():
        shutil.copyfile(source, target)
    return target


def process_run(run_dir, dataset_root, device, num_samples, grid, force,
                amp=False, cond_batch=1, limit=0, include_best=True,
                metrics_filename="metrics.csv"):
    run_dir = Path(run_dir)
    run_name = run_dir.name
    if not (run_dir / "metrics.csv").exists():
        print("[skip] {}: no metrics.csv".format(run_name), flush=True)
        return None
    metrics_path = prepare_metrics_path(run_dir, metrics_filename)

    # Reference metadata comes from the requested grid in strict mode so that
    # best.pt is neither loaded nor sampled.
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    if include_best and not best_path.exists():
        print("[skip] {}: no best.pt".format(run_name), flush=True)
        return None
    if include_best:
        reference_path = best_path
    else:
        reference_path = run_dir / "checkpoints" / "epoch_{:04d}.pt".format(int(grid[0]))
        if not reference_path.exists():
            raise FileNotFoundError(
                "Required grid checkpoint missing for {}: {}".format(
                    run_name, reference_path.name
                )
            )
    model, config, reference_epoch = load_checkpoint(reference_path, device)
    best_epoch = reference_epoch
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    reference_label = "best.pt" if include_best else reference_path.name
    print("== {} ==  (reference {} epoch {}, T={}, S={})".format(
        run_name, reference_label, reference_epoch, config.timesteps, config.ddim_steps), flush=True)

    val_data = load_val_data(dataset_root, config)
    if limit and limit > 0:
        val_data = val_data[:int(limit)]
        print("  [limit] evaluating first {} validation pairs only".format(len(val_data)), flush=True)

    ckpt_jobs = _checkpoint_jobs(
        run_dir, grid, best_epoch=reference_epoch, include_best=include_best
    )
    if not include_best:
        missing = [
            run_dir / "checkpoints" / "epoch_{:04d}.pt".format(epoch)
            for epoch in grid
            if not (run_dir / "checkpoints" / "epoch_{:04d}.pt".format(epoch)).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Required grid checkpoint(s) missing for {}: {}".format(
                    run_name, ", ".join(path.name for path in missing)
                )
            )

    rows, columns = read_metrics_csv(metrics_path)
    done_epochs = set()
    for row in rows:
        if row.get("ssim", "") not in ("", None):
            try:
                done_epochs.add(int(float(row.get("epoch", -1))))
            except ValueError:
                pass

    # epoch -> (prob_maps, sample0) cache, only kept for this run
    cache = {}
    for epoch, path in ckpt_jobs:
        if epoch in done_epochs and not force:
            print("  epoch {:4d}: already evaluated, skip".format(epoch), flush=True)
            continue
        start = time.time()
        model, ckpt_config, ckpt_epoch = load_checkpoint(path, device)
        assert ckpt_epoch == epoch, "epoch mismatch for {}: {} vs {}".format(path, ckpt_epoch, epoch)
        prob_maps, sample0, means = evaluate_checkpoint(
            model, ckpt_config, val_data, num_samples, device, config.seed,
            amp=amp, cond_batch=cond_batch,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        cache[epoch] = (prob_maps, sample0, ckpt_config.ddim_steps)
        merge_epoch_metrics(metrics_path, epoch, means)
        if not include_best:
            output_name = "prediction_epoch_{:04d}_samples{:03d}".format(
                epoch, num_samples
            )
            save_keep_set(
                run_dir,
                output_name,
                epoch,
                prob_maps,
                sample0,
                val_data,
                num_samples,
                ckpt_config.ddim_steps,
            )
        elapsed = time.time() - start
        print("  epoch {:4d}: ssim={:.4f} dice={:.4f} iou={:.4f} orient={:.4f} "
              "outside={:.4f}  ({:.1f}s)".format(
                  epoch, means["ssim"], means["dice"], means["iou"],
                  means["orient"], means["outside_mean"], elapsed), flush=True)

    # recompute score_avg over all evaluated epochs
    fill_score_avg(metrics_path)

    # Targeted grid jobs must stop here.  The normal keep-set stage can
    # regenerate historical winner checkpoints (including best.pt), which is
    # intentionally excluded from a strict grid-only evaluation.
    if not include_best:
        print("  grid-only evaluation complete; skipping best/keep-set sampling", flush=True)
        return metrics_path

    # decide keep-sets from the merged table
    rows, columns = read_metrics_csv(metrics_path)
    evaluated = []
    for row in rows:
        if row.get("score_avg", "") in ("", None):
            continue
        try:
            evaluated.append((int(float(row.get("epoch", 0))), row))
        except ValueError:
            continue
    if not evaluated:
        print("  [warn] no evaluated epochs; skip keep-sets and plots", flush=True)
        return metrics_path

    def winner(metric):
        valid = [(ep, float(row[metric])) for ep, row in evaluated
                 if row.get(metric, "") not in ("", None)]
        if not valid:
            print("  [warn] metric {} has no evaluated values; role skipped".format(metric),
                  flush=True)
            return None
        return max(valid, key=lambda t: t[1])[0]

    def winner_avg():
        return max(evaluated, key=lambda t: float(t[1]["score_avg"]))[0]

    last_epoch = None
    if last_path.exists():
        _, _, last_epoch = load_checkpoint(last_path, device)
    else:
        last_epoch = max(ep for ep, _ in evaluated)

    roles = []
    for metric in KEEP_ROLES_METRIC:
        roles.append(("metric_" + metric, "prediction_metricbest_" + metric, winner(metric)))
    roles.append(("avg", KEEP_ROLES_SPECIAL[0][1], winner_avg()))
    roles.append(("bestpt", KEEP_ROLES_SPECIAL[1][1], best_epoch))
    roles.append(("lastpt", KEEP_ROLES_SPECIAL[2][1], last_epoch))

    keep_summary = {"run": run_name, "num_samples": num_samples, "roles": {}}
    saved_epochs = {}
    for role, folder, epoch in roles:
        if epoch is None:
            keep_summary["roles"][role] = {"epoch": None, "folder": None,
                                           "note": "no evaluated values for this role"}
            continue
        if epoch in cache:
            prob_maps, sample0, ddim_steps = cache[epoch]
        else:
            # winner was evaluated in a previous session: regenerate once
            path = run_dir / "checkpoints" / "epoch_{:04d}.pt".format(epoch)
            if not path.exists():
                path = best_path if epoch == best_epoch else None
            if path is None or not path.exists():
                keep_summary["roles"][role] = {"epoch": epoch, "folder": None,
                                               "note": "checkpoint missing"}
                continue
            model, ckpt_config, _ = load_checkpoint(path, device)
            prob_maps, sample0, ddim_steps = evaluate_checkpoint(
                model, ckpt_config, val_data, num_samples, device, config.seed,
                amp=amp, cond_batch=cond_batch,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if epoch in saved_epochs:
            keep_summary["roles"][role] = {"epoch": epoch,
                                           "folder": saved_epochs[epoch],
                                           "note": "same checkpoint as role folder {}".format(
                                               saved_epochs[epoch])}
            continue
        save_keep_set(run_dir, folder, epoch, prob_maps, sample0, val_data,
                      num_samples, ddim_steps)
        saved_epochs[epoch] = folder
        keep_summary["roles"][role] = {"epoch": epoch, "folder": folder}
        print("  keep-set {} -> epoch {} saved as {}".format(role, epoch, folder), flush=True)

    (run_dir / "keep_summary.json").write_text(
        json.dumps(keep_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    plot_run_trend(run_dir, metrics_path, best_epoch, run_name)
    print("  trend plot saved", flush=True)
    return metrics_path


def main():
    parser = argparse.ArgumentParser(description="Matrix checkpoint screening evaluation")
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--grid", default=",".join(str(g) for g in DEFAULT_GRID))
    parser.add_argument("--skip", default="lr_high",
                        help="comma-separated substrings of run names to skip")
    parser.add_argument("--only", default="",
                        help="comma-separated substrings; only matching run directories")
    parser.add_argument("--exact_run", default="",
                        help="evaluate exactly one run directory name")
    parser.add_argument("--metrics_filename", default="metrics.csv",
                        help="per-run metrics CSV filename (default: metrics.csv)")
    parser.add_argument("--no_summary", action="store_true",
                        help="skip the cross-run summary plot (use on parallel workers)")
    parser.add_argument("--force", action="store_true",
                        help="re-evaluate epochs that already have scores")
    parser.add_argument("--grid_only", action="store_true",
                        help="evaluate only --grid checkpoints; never load or evaluate best.pt")
    parser.add_argument("--amp", action="store_true",
                        help="use automatic mixed precision for sampling (faster, screening only)")
    parser.add_argument("--cond_batch", type=int, default=1,
                        help="how many conditions to pack into one sampling batch")
    parser.add_argument("--limit", type=int, default=0,
                        help="evaluate only the first N validation pairs (0 = all)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    grid = [int(item) for item in args.grid.split(",") if item.strip()]
    skip_tokens = [token.strip() for token in args.skip.split(",") if token.strip()]
    only_tokens = [token.strip() for token in args.only.split(",") if token.strip()]

    runs_root = Path(args.runs_root)
    run_dirs = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        if args.exact_run and child.name != args.exact_run:
            continue
        if any(token in child.name for token in skip_tokens):
            print("[skip-by-name] {}".format(child.name), flush=True)
            continue
        if only_tokens and not any(token in child.name for token in only_tokens):
            continue
        if args.grid_only:
            has_grid = all(
                (child / "checkpoints" / "epoch_{:04d}.pt".format(epoch)).exists()
                for epoch in grid
            )
            if not has_grid:
                continue
            if not (child / "metrics.csv").exists():
                continue
        elif not (child / "best.pt").exists():
            continue
        run_dirs.append(child)
    print("runs to evaluate: {}".format([d.name for d in run_dirs]), flush=True)
    if args.exact_run and len(run_dirs) != 1:
        raise SystemExit("Requested run not found or missing checkpoint: {}".format(args.exact_run))

    run_results = []
    failed_runs = []
    for run_dir in run_dirs:
        try:
            metrics_path = process_run(run_dir, args.dataset_root, device,
                                       args.num_samples, grid, args.force,
                                       amp=args.amp, cond_batch=args.cond_batch,
                                       limit=args.limit,
                                       include_best=not args.grid_only,
                                       metrics_filename=args.metrics_filename)
            if metrics_path:
                run_results.append((run_dir.name, metrics_path))
        except Exception:
            failed_runs.append(run_dir.name)
            print("[error] run {} failed:".format(run_dir.name), flush=True)
            import traceback
            traceback.print_exc()
            print("continuing with next run...", flush=True)

    if run_results and not args.no_summary:
        plot_matrix_summary(runs_root, run_results)
        print("summary plot saved to {}".format(runs_root / "evaluation_summary.png"), flush=True)
    if failed_runs:
        print("evaluation failed for run(s): {}".format(", ".join(failed_runs)), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
