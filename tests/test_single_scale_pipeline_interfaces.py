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
sp = importlib.import_module("vessel_reproduction.single_scale_pipeline")


class TestSingleScalePipelineInterfaces(unittest.TestCase):
    def test_required_symbols_are_exported(self) -> None:
        names = {
            "AdaptiveThresholdOptions",
            "AreaDenoiseOptions",
            "PipelineDebugOptions",
            "SingleScalePipelineParams",
            "PipelineConfigError",
            "PipelineInputError",
            "validate_threshold_options",
            "validate_denoise_options",
            "validate_pipeline_input",
            "adaptive_threshold_single",
            "remove_small_components",
            "run_single_scale_pipeline",
        }
        for name in names:
            self.assertTrue(hasattr(vr, name), f"missing export: {name}")

    def test_threshold_and_denoise(self) -> None:
        img = np.random.rand(32, 32).astype(np.float32)
        thr = sp.adaptive_threshold_single(img, options=sp.AdaptiveThresholdOptions())
        self.assertEqual(thr.shape, img.shape)

        den = sp.remove_small_components(thr, options=sp.AreaDenoiseOptions(area_min=3))
        self.assertEqual(den.shape, img.shape)

    def test_pipeline_end_to_end(self) -> None:
        image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        params = sp.SingleScalePipelineParams()
        mask, inter = sp.run_single_scale_pipeline(image, params)
        self.assertEqual(mask.shape, image.shape)
        self.assertIn("illumination_corrected", inter)
        self.assertIn("tophat_fused", inter)
        self.assertIn("threshold_binary", inter)
        self.assertIn("denoised_binary", inter)

    def test_main_signature(self) -> None:
        sig = inspect.signature(sp.run_single_scale_pipeline)
        self.assertIn("image", sig.parameters)
        self.assertIn("params", sig.parameters)


if __name__ == "__main__":
    unittest.main()
