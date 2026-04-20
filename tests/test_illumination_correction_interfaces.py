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
ic = importlib.import_module("vessel_reproduction.illumination_correction")


class TestIlluminationCorrectionInterfaces(unittest.TestCase):
    def test_required_symbols_are_exported(self) -> None:
        names = {
            "IlluminationCorrectionOptions",
            "IlluminationConfigError",
            "IlluminationInputError",
            "validate_illumination_params",
            "normalize_input_image",
            "apply_mean_filter",
            "illumination_correction",
        }
        for name in names:
            self.assertTrue(hasattr(vr, name), f"missing export: {name}")

    def test_validate_params_constraints(self) -> None:
        ic.validate_illumination_params(window_size=7, c=0.01, eps=1e-6)
        with self.assertRaises(ic.IlluminationConfigError):
            ic.validate_illumination_params(window_size=2, c=0.01)

    def test_functions_output_shape(self) -> None:
        image = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        norm = ic.normalize_input_image(image, in_range="auto")
        self.assertEqual(norm.shape, image.shape)
        self.assertEqual(norm.dtype, np.float32)

        mean = ic.apply_mean_filter(norm, window_size=7)
        self.assertEqual(mean.shape, image.shape)

        out = ic.illumination_correction(image, window_size=7, c=0.01)
        self.assertEqual(out.shape, image.shape)

    def test_image_shape_constraints(self) -> None:
        with self.assertRaises(ic.IlluminationInputError):
            ic.illumination_correction(np.zeros((4, 4, 3), dtype=np.uint8), window_size=7, c=0.01)

    def test_main_signature(self) -> None:
        sig = inspect.signature(ic.illumination_correction)
        self.assertIn("image", sig.parameters)
        self.assertIn("window_size", sig.parameters)


if __name__ == "__main__":
    unittest.main()
