# Actin Stress Fiber Diffusion Training

## Scope

This training path implements the conditional diffusion objective described by Li et al. (2024):

- 256x256 grayscale cell contour condition and stress-fiber target.
- Cosine variance schedule with `T=800`.
- Epsilon prediction with MSE.
- Adam with learning rate `1.5e-4`.
- Batch size `16`.
- `1500` epochs for the paper-aligned baseline.
- Random rotations by multiples of 90 degrees and horizontal/vertical flips.
- DDIM generation with `S=80`.
- One condition is evaluated with `100` generated samples and an averaged probability map.

The paper does not define a validation checkpoint rule, seed, AMP, gradient clipping, or exact U-Net width. Those are explicit engineering defaults and are recorded in `config.json` for every run.

## Data layout

The preferred extracted data layout is:

```text
dataset/
  contour/
    0000.jpg
  SF/
    0000.jpg
```

This is the directory layout provided by the generator dataset. `contour` is the condition image and `SF` is the stress-fiber target. The condition and target must have the same relative filename stem. The loader sorts and matches by stem; it never relies on directory enumeration order. If `validation/` is absent, the 1293 pairs are split deterministically using `--val_fraction`.

This code does not extract `.7z` archives. Extract the dataset before training and pass the extracted directory.

## Environment

The environment declaration from the original README is documented in `ENVIRONMENT.md`.
The Conda configuration is provided in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate actin-stress-fibers-generator
```

The U-Net uses `einops` when available. `rearrange_compat.py` contains a local fallback for the exact patterns used by this repository.

## Single baseline run

From this project directory:

```bash
python train_diffusion.py \
  --output_dir runs/paper_baseline \
  --device cuda
```

默认使用单卡。双卡训练使用同一个入口，显式打开 `DataParallel`：

```bash
python train_diffusion.py \
  --output_dir runs/paper_baseline_2gpu \
  --device cuda:0 \
  --use_data_parallel true \
  --gpu_ids 0,1
```

也可以直接在 `TrainingConfig` 中设置 `use_data_parallel=True` 和 `gpu_ids=(0, 1)`。
双卡模式要求 CUDA 可用且指定 GPU 编号存在；单卡模式保持原有训练行为。

When executed from the generator project directory, `--dataset_root` defaults to `dataset`.
An explicit path can still be supplied:

```bash
python train_diffusion.py \
  --dataset_root /path/to/dataset \
  --output_dir runs/paper_baseline \
  --device cuda
```

The run writes:

```text
runs/paper_baseline/
  config.json
  metrics.csv
  last.pt
  best.pt
  checkpoints/
  samples/
```

Resume from the latest checkpoint:

```bash
python train_diffusion.py \
  --dataset_root /path/to/dataset \
  --output_dir runs/paper_baseline \
  --resume runs/paper_baseline/last.pt \
  --device cuda
```

## Repeated-sample evaluation

After training, generate 100 samples per condition and write an averaged probability map:

```bash
python evaluate_diffusion.py \
  --checkpoint runs/paper_baseline/best.pt \
  --condition_dir /path/to/dataset/validation/condition \
  --output_dir runs/paper_baseline/evaluation \
  --num_samples 100 \
  --device cuda
```

Each condition gets a numbered directory containing `probability_map.npy`, `probability_map.png`, the condition image, and one representative sample. `metadata.json` records the sampling settings.

The probability map follows the paper's procedure: each generated grayscale output is thresholded at the normalized equivalent of 128/255, then binary maps are averaged.

## One-command batch training

The default matrix contains the paper baseline and focused ablations. Each experiment is executed sequentially and writes to its own directory:

```bash
bash scripts/run_batch_train.sh /path/to/dataset /path/to/runs
```

To use a custom matrix:

```bash
bash scripts/run_batch_train.sh /path/to/dataset /path/to/runs configs/experiment_matrix.json
```

For a server with multiple GPUs, select one GPU in the shell environment:

```bash
CUDA_VISIBLE_DEVICES=0 DEVICE=cuda bash scripts/run_batch_train.sh /path/to/dataset /path/to/runs
```

On Windows PowerShell:

```powershell
.\scripts\run_batch_train.ps1 -DatasetRoot D:\data\dataset -OutputRoot D:\runs
```

## Hyperparameters worth adjusting

Adjust these first because they directly affect convergence or generation quality:

1. `lr`: start with `7.5e-5`, `1.5e-4`, and `3e-4`.
2. `batch_size`: use `16` for the paper baseline; use `8` if memory is insufficient. If changing batch size, compare effective batch size before interpreting the result.
3. `timesteps` and `ddim_steps`: compare `(400, 40)`, `(800, 80)`, and `(1200, 120)`.
4. `augmentation_prob`: compare `1.0` and `0.0`; keep rotations/flips synchronized between condition and target.
5. `loss_type`: keep MSE as the primary result; use Huber only as a robustness ablation.
6. `model_dim` and `dim_mults`: only tune after the optimization/data ablations. The default follows the existing six-stage U-Net and is memory intensive at 256x256.

Do not change normalization, filename pairing, or DDIM timestep indexing between experiments. Those are correctness conditions, not tunable hyperparameters.

## Recommended matrix

| Run | Learning rate | Batch | T | DDIM S | Loss | Augmentation |
|---|---:|---:|---:|---:|---|---:|
| `paper_baseline` | 1.5e-4 | 16 | 800 | 80 | MSE | on |
| `lr_low` | 7.5e-5 | 16 | 800 | 80 | MSE | on |
| `lr_high` | 3.0e-4 | 16 | 800 | 80 | MSE | on |
| `timesteps_400` | 1.5e-4 | 16 | 400 | 40 | MSE | on |
| `timesteps_1200` | 1.5e-4 | 16 | 1200 | 120 | MSE | on |
| `batch_8` | 1.5e-4 | 8 | 800 | 80 | MSE | on |
| `huber_loss` | 1.5e-4 | 16 | 800 | 80 | Huber | on |
| `no_augmentation` | 1.5e-4 | 16 | 800 | 80 | MSE | off |

Use validation epsilon MSE to detect optimization failure, but select the final model with the paper-aligned repeated-sample metrics: 64x64 grid ROC/AUC, total SF length error, principal direction error, and order parameter agreement. The training objective alone does not measure these geometry properties.
