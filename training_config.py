import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("Expected a boolean value, got {!r}".format(value))


def parse_dim_mults(value):
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("dim_mults must contain at least one integer")
    return tuple(int(item) for item in items)


def parse_gpu_ids(value):
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("gpu_ids must contain at least one integer")
    return tuple(int(item) for item in items)


@dataclass
class TrainingConfig:
    dataset_root: Optional[str] = "dataset"
    condition_dir: Optional[str] = None
    target_dir: Optional[str] = None
    val_condition_dir: Optional[str] = None
    val_target_dir: Optional[str] = None
    output_dir: str = "runs/paper_baseline"
    resolution: int = 256
    batch_size: int = 16
    epochs: int = 1500
    timesteps: int = 800
    ddim_steps: int = 80
    lr: float = 1.5e-4
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    loss_type: str = "mse"
    augmentation_prob: float = 1.0
    val_fraction: float = 0.1
    seed: int = 42
    num_workers: int = 4
    device: str = "auto"
    use_data_parallel: bool = False
    gpu_ids: Tuple[int, ...] = (0, 1)
    amp: bool = True
    grad_clip: float = 1.0
    model_dim: int = 32
    dim_mults: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    groups: int = 8
    sample_every: int = 25
    sample_count: int = 4
    save_every: int = 25
    resume: Optional[str] = None
    max_train_batches: int = 0
    max_val_batches: int = 0

    def __post_init__(self):
        self.dim_mults = parse_dim_mults(self.dim_mults)
        self.gpu_ids = parse_gpu_ids(self.gpu_ids)
        self.validate()

    def validate(self):
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("batch_size and epochs must be positive")
        if self.timesteps <= 1:
            raise ValueError("timesteps must be greater than one")
        if self.ddim_steps <= 0 or self.ddim_steps > self.timesteps:
            raise ValueError("ddim_steps must be in [1, timesteps]")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= self.augmentation_prob <= 1:
            raise ValueError("augmentation_prob must be in [0, 1]")
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be in (0, 1)")
        if self.model_dim <= 0 or self.groups <= 0:
            raise ValueError("model_dim and groups must be positive")
        if self.model_dim % self.groups != 0:
            raise ValueError("model_dim must be divisible by groups")
        if len(self.dim_mults) == 0:
            raise ValueError("dim_mults cannot be empty")
        downsample_factor = 2 ** (len(self.dim_mults) - 1)
        if self.resolution % downsample_factor != 0:
            raise ValueError(
                "resolution must be divisible by {} for dim_mults {}".format(
                    downsample_factor, self.dim_mults
                )
            )
        if self.loss_type not in {"mse", "l1", "huber"}:
            raise ValueError("loss_type must be one of mse, l1, huber")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if any(item < 0 for item in self.gpu_ids):
            raise ValueError("gpu_ids cannot contain negative values")
        if self.use_data_parallel and len(self.gpu_ids) < 2:
            raise ValueError("use_data_parallel requires at least two gpu_ids")
        if self.sample_every < 0 or self.save_every < 0:
            raise ValueError("sample_every and save_every cannot be negative")

    def to_dict(self):
        values = asdict(self)
        values["dim_mults"] = list(self.dim_mults)
        return values

    @classmethod
    def from_dict(cls, values):
        return cls(**dict(values))

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Train the conditional actin diffusion model")
    parser.add_argument("--dataset_root", default="dataset")
    parser.add_argument("--condition_dir")
    parser.add_argument("--target_dir")
    parser.add_argument("--val_condition_dir")
    parser.add_argument("--val_target_dir")
    parser.add_argument("--output_dir", default="runs/paper_baseline")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--timesteps", type=int, default=800)
    parser.add_argument("--ddim_steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--loss_type", choices=("mse", "l1", "huber"), default="mse")
    parser.add_argument("--augmentation_prob", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use_data_parallel", type=parse_bool, default=False)
    parser.add_argument("--gpu_ids", type=parse_gpu_ids, default=(0, 1))
    parser.add_argument("--amp", type=parse_bool, default=True)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--model_dim", type=int, default=32)
    parser.add_argument("--dim_mults", type=parse_dim_mults, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--sample_every", type=int, default=25)
    parser.add_argument("--sample_count", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--resume")
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    return parser


def config_from_args(args=None):
    namespace = build_parser().parse_args(args)
    return TrainingConfig(**vars(namespace))
