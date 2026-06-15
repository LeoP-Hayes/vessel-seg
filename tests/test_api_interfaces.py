from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import vessel_reproduction as vr


class TestApiInterfaces(unittest.TestCase):
    def test_symbols_exported(self) -> None:
        self.assertTrue(hasattr(vr, "BatchExtractionSummary"))
        self.assertTrue(hasattr(vr, "extract_vessel_mask"))
        self.assertTrue(hasattr(vr, "extract_vessel_masks_batch"))

    def test_extract_vessel_mask_single(self) -> None:
        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        mask = vr.extract_vessel_mask(image, pipeline="single", config_path=None)
        self.assertEqual(mask.shape, image.shape)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertTrue(set(np.unique(mask)).issubset({0, 1}))

    def test_extract_vessel_masks_batch_train_val_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_root = tmp / "input"
            output_root = tmp / "output"
            for split in ("train", "val", "test"):
                split_b = input_root / split / "B"
                split_b.mkdir(parents=True, exist_ok=True)
                img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
                cv2.imwrite(str(split_b / f"{split}_0.png"), img)

            summary = vr.extract_vessel_masks_batch(
                input_root,
                output_root,
                split_mode="train_val_test",
                pipeline="single",
                config_path=None,
                on_error="raise",
            )

            self.assertEqual(summary.total, 3)
            self.assertEqual(summary.processed, 3)
            self.assertEqual(summary.failed, 0)
            for split in ("train", "val", "test"):
                out = output_root / split / "B_mask" / f"{split}_0.png"
                self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
