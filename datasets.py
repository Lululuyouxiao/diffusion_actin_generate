import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImagePair:
    key: str
    condition: Path
    target: Path


def _image_files(root: Path):
    if not root.exists():
        raise FileNotFoundError("Image directory does not exist: {}".format(root))
    if not root.is_dir():
        raise NotADirectoryError("Expected an image directory: {}".format(root))
    return {
        path.relative_to(root).with_suffix("").as_posix().lower(): path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def pair_image_files(condition_dir, target_dir) -> List[ImagePair]:
    condition_root = Path(condition_dir)
    target_root = Path(target_dir)
    condition_files = _image_files(condition_root)
    target_files = _image_files(target_root)
    condition_keys = set(condition_files)
    target_keys = set(target_files)
    missing_targets = sorted(condition_keys - target_keys)
    missing_conditions = sorted(target_keys - condition_keys)
    if missing_targets or missing_conditions:
        details = []
        if missing_targets:
            details.append("missing targets: {}".format(", ".join(missing_targets[:10])))
        if missing_conditions:
            details.append("missing conditions: {}".format(", ".join(missing_conditions[:10])))
        raise ValueError("Condition/target pairing failed; " + "; ".join(details))
    if not condition_files:
        raise ValueError("No supported images found in {}".format(condition_root))
    return [
        ImagePair(key=key, condition=condition_files[key], target=target_files[key])
        for key in sorted(condition_files)
    ]


def split_pairs(pairs: Sequence[ImagePair], val_fraction: float, seed: int):
    if len(pairs) < 2:
        raise ValueError("At least two paired images are required for a train/validation split")
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(round(len(indices) * val_fraction)))
    val_indices = set(indices[:val_size])
    train = [pair for index, pair in enumerate(pairs) if index not in val_indices]
    validation = [pair for index, pair in enumerate(pairs) if index in val_indices]
    return train, validation


def _find_subdirectory(root: Path, names: Iterable[str]) -> Optional[Path]:
    candidates = {name.lower() for name in names}
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() in candidates:
            return child
    return None


def discover_split_directories(dataset_root, split):
    root = Path(dataset_root)
    split_root = root / split if (root / split).is_dir() else root
    condition = _find_subdirectory(
        split_root,
        (
            "condition",
            "conditions",
            "contour",
            "contours",
            "input",
            "inputs",
            "cell",
            "cells",
            "morphology",
        ),
    )
    target = _find_subdirectory(
        split_root,
        ("target", "targets", "output", "outputs", "sf", "stress_fibers", "stress-fibers"),
    )
    if condition is None or target is None:
        raise FileNotFoundError(
            "Could not find condition/target directories under {}".format(split_root)
        )
    return condition, target


def _to_tensor(image: Image.Image, resolution: int):
    image = image.convert("L")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = image.resize((resolution, resolution), resampling)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).unsqueeze(0)


def _spatial_augment(condition, target):
    rotation = random.randrange(4)
    if rotation:
        condition = condition.rotate(90 * rotation)
        target = target.rotate(90 * rotation)
    transpose = getattr(Image, "Transpose", Image)
    if random.random() < 0.5:
        condition = condition.transpose(transpose.FLIP_LEFT_RIGHT)
        target = target.transpose(transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        condition = condition.transpose(transpose.FLIP_TOP_BOTTOM)
        target = target.transpose(transpose.FLIP_TOP_BOTTOM)
    return condition, target


class PairedImageDataset(Dataset):
    def __init__(self, pairs: Sequence[ImagePair], resolution=256, augment=False, augmentation_prob=1.0):
        self.pairs = list(pairs)
        self.resolution = int(resolution)
        self.augment = bool(augment)
        self.augmentation_prob = float(augmentation_prob)
        if not self.pairs:
            raise ValueError("PairedImageDataset cannot be empty")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        condition = Image.open(pair.condition).convert("L")
        target = Image.open(pair.target).convert("L")
        if self.augment and random.random() < self.augmentation_prob:
            condition, target = _spatial_augment(condition, target)
        return _to_tensor(condition, self.resolution), _to_tensor(target, self.resolution)


class ConditionImageDataset(Dataset):
    def __init__(self, condition_dir, resolution=256):
        self.paths = [
            path
            for path in sorted(Path(condition_dir).rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not self.paths:
            raise ValueError("No supported condition images found in {}".format(condition_dir))
        self.resolution = int(resolution)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        return _to_tensor(Image.open(path), self.resolution), path.name
