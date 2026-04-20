from __future__ import annotations

import importlib
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import vessel_reproduction as vr
dp = importlib.import_module("vessel_reproduction.dual_scale_pipeline")


class TestDualScalePipelineInterfaces(unittest.TestCase):
    def test_required_symbols_are_exported(self) -> None:
        names = {
            "AlignOptions",
            "MultiScaleDebugOptions",
            "DualScalePipelineParams",
            "DualScaleConfigError",
            "DualScaleInputError",
            "validate_dual_scale_input",
            "validate_dual_scale_params",
            "build_single_scale_params_for_level",
            "downsample_image",
            "upsample_binary_mask_to_shape",
            "fuse_masks_or",
            "run_dual_scale_pipeline",
        }
        for name in names:
            self.assertTrue(hasattr(vr, name), f"missing export: {name}")

    def test_downsample_and_upsample(self) -> None:
        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        small = dp.downsample_image(image, scale=0.5)
        self.assertEqual(small.shape, (32, 32))

        up = dp.upsample_binary_mask_to_shape((small > 128).astype(np.uint8), target_shape=(64, 64))
        self.assertEqual(up.shape, (64, 64))

    def test_fuse_or_logic(self) -> None:
        a = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        b = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        fused = dp.fuse_masks_or(a, b, output_dtype="uint8")
        self.assertTrue(np.array_equal(fused, np.array([[1, 1], [0, 0]], dtype=np.uint8)))

    def test_run_dual_scale_pipeline(self) -> None:
        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        params = dp.DualScalePipelineParams()
        mask, inter = dp.run_dual_scale_pipeline(image, params)
        self.assertEqual(mask.shape, image.shape)
        self.assertIn("fused_or", inter)

    def test_main_signature(self) -> None:
        sig = inspect.signature(dp.run_dual_scale_pipeline)
        self.assertIn("image", sig.parameters)
        self.assertIn("params", sig.parameters)


if __name__ == "__main__":
    unittest.main()
