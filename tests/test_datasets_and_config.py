import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from datasets import (
    PairedImageDataset,
    discover_split_directories,
    pair_image_files,
)
from training_config import config_from_args, parse_bool


class DatasetAndConfigTests(unittest.TestCase):
    def _write_image(self, path, value, size=(4, 2)):
        array = np.full((size[1], size[0]), value, dtype=np.uint8)
        Image.fromarray(array).save(path)

    def test_parse_bool_rejects_ambiguous_values(self):
        self.assertTrue(parse_bool("true"))
        self.assertFalse(parse_bool("False"))
        with self.assertRaises(ValueError):
            parse_bool("not-a-bool")

    def test_training_defaults_to_original_generator_dataset_directory(self):
        self.assertEqual(config_from_args([]).dataset_root, "dataset")

    def test_pairs_are_matched_by_relative_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition = root / "condition"
            target = root / "target"
            condition.mkdir()
            target.mkdir()
            self._write_image(condition / "b.png", 10)
            self._write_image(condition / "a.png", 20)
            self._write_image(target / "a.png", 30)
            self._write_image(target / "b.png", 40)

            pairs = pair_image_files(condition, target)

            self.assertEqual([pair.key for pair in pairs], ["a", "b"])
            self.assertEqual(pairs[0].condition.name, "a.png")
            self.assertEqual(pairs[0].target.name, "a.png")

    def test_missing_pair_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition = root / "condition"
            target = root / "target"
            condition.mkdir()
            target.mkdir()
            self._write_image(condition / "only_condition.png", 10)

            with self.assertRaises(ValueError):
                pair_image_files(condition, target)

    def test_dataset_resizes_and_normalizes_to_minus_one_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            condition = root / "condition"
            target = root / "target"
            condition.mkdir()
            target.mkdir()
            self._write_image(condition / "sample.png", 0, size=(4, 2))
            self._write_image(target / "sample.png", 255, size=(4, 2))

            dataset = PairedImageDataset(
                pair_image_files(condition, target),
                resolution=8,
                augment=False,
            )
            condition_tensor, target_tensor = dataset[0]

            self.assertEqual(tuple(condition_tensor.shape), (1, 8, 8))
            self.assertEqual(tuple(target_tensor.shape), (1, 8, 8))
            self.assertAlmostEqual(float(condition_tensor.min()), -1.0, places=6)
            self.assertAlmostEqual(float(target_tensor.max()), 1.0, places=6)

    def test_original_generator_dataset_layout_uses_contour_and_sf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            contour = root / "contour"
            stress_fibers = root / "SF"
            contour.mkdir()
            stress_fibers.mkdir()
            self._write_image(contour / "0000.jpg", 10)
            self._write_image(stress_fibers / "0000.jpg", 20)

            condition_dir, target_dir = discover_split_directories(root, "train")
            pairs = pair_image_files(condition_dir, target_dir)

            self.assertEqual(condition_dir.name, "contour")
            self.assertEqual(target_dir.name, "SF")
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].key, "0000")


if __name__ == "__main__":
    unittest.main()
