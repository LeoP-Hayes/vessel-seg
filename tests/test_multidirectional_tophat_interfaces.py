from __future__ import annotations

import importlib
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import vessel_reproduction as vr
mt = importlib.import_module("vessel_reproduction.multidirectional_tophat")


class TestMultidirectionalTopHatInterfaces(unittest.TestCase):
    def test_required_symbols_are_exported(self) -> None:
        names = {
            "TopHatOptions",
            "DirectionSpec",
            "TopHatIntermediate",
            "TopHatConfigError",
            "TopHatInputError",
            "validate_tophat_params",
            "generate_direction_specs",
            "generate_line_structuring_element",
            "compute_modified_tophat_response",
            "fuse_directional_responses",
            "run_multidirectional_tophat",
            "serialize_tophat_intermediate",
            "intermediate_to_rows",
        }
        for name in names:
            self.assertTrue(hasattr(vr, name), f"missing export: {name}")

    def test_direction_specs_m9_endpoint(self) -> None:
        specs = mt.generate_direction_specs(num_directions=9, start_deg=0.0, end_deg=180.0, include_endpoint=True)
        got = [round(s.angle_deg, 4) for s in specs]
        self.assertEqual(got, [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0])

    def test_compute_and_run(self) -> None:
        image = np.random.rand(32, 32).astype(np.float32)
        se = mt.generate_line_structuring_element(line_length=7, angle_deg=45.0)
        resp, bg = mt.compute_modified_tophat_response(image, selem=se)
        self.assertEqual(resp.shape, image.shape)
        self.assertEqual(bg.shape, image.shape)

        fused, inter = mt.run_multidirectional_tophat(image, num_directions=9, line_length=6, save_intermediate=True, save_directional=True)
        self.assertEqual(fused.shape, image.shape)
        self.assertIsNotNone(inter)
        assert inter is not None
        self.assertEqual(len(inter.angles_deg), 9)

    def test_serialize_and_rows(self) -> None:
        image = np.random.rand(16, 16).astype(np.float32)
        fused, inter = mt.run_multidirectional_tophat(image, num_directions=3, line_length=5, save_intermediate=True, save_directional=True)
        assert inter is not None

        with tempfile.TemporaryDirectory() as td:
            saved = mt.serialize_tophat_intermediate(inter, output_dir=td, prefix="demo", save_directional=True)
            self.assertGreaterEqual(len(saved), 1)
            for p in saved:
                self.assertTrue(Path(p).exists())

        rows = mt.intermediate_to_rows(inter)
        self.assertGreaterEqual(len(rows), 1)

    def test_main_signature(self) -> None:
        sig = inspect.signature(mt.run_multidirectional_tophat)
        self.assertIn("image", sig.parameters)
        self.assertIn("num_directions", sig.parameters)


if __name__ == "__main__":
    unittest.main()
