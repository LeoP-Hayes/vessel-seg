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
nb = importlib.import_module("vessel_reproduction.notebook_debug")


class TestNotebookDebugInterfaces(unittest.TestCase):
    def test_required_symbols_are_exported(self) -> None:
        names = {
            "NOTEBOOK_SECTIONS",
            "DEFAULT_PANELS",
            "NotebookRunConfig",
            "NotebookConfigError",
            "NotebookInputError",
            "make_overlay",
            "run_once",
            "compare_param_sets",
            "show_debug_panels",
            "show_comparison_grid",
            "show_metrics_table",
            "export_notebook_artifacts",
        }
        for name in names:
            self.assertTrue(hasattr(vr, name), f"missing export: {name}")

    def test_sections_and_panels(self) -> None:
        self.assertEqual(nb.NOTEBOOK_SECTIONS[0], "00_setup")
        self.assertIn("04_visualization_panels", nb.NOTEBOOK_SECTIONS)
        self.assertIn("overlay", nb.DEFAULT_PANELS)

    def test_overlay_output(self) -> None:
        image = np.zeros((8, 8), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:4, 2:4] = 1
        overlay = nb.make_overlay(image, mask, mask_mode="binary01", alpha=0.5)
        self.assertEqual(overlay.shape, (8, 8, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_overlay_constraints(self) -> None:
        with self.assertRaises(nb.NotebookInputError):
            nb.make_overlay(np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8, 1), dtype=np.uint8))
        with self.assertRaises(nb.NotebookConfigError):
            nb.make_overlay(np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8), alpha=1.5)

    def test_stub_functions_raise_not_implemented(self) -> None:
        run_cfg = nb.NotebookRunConfig(sample_path="data/test/0.png")
        with self.assertRaises(NotImplementedError):
            nb.run_once(run_cfg, pipeline_params={})

        image = np.zeros((8, 8), dtype=np.uint8)
        with self.assertRaises(NotImplementedError):
            nb.compare_param_sets(image, [("base", {})])

        with self.assertRaises(NotImplementedError):
            nb.show_debug_panels({"intermediate": {}}, show_directional=False)

        with self.assertRaises(NotImplementedError):
            nb.show_comparison_grid([{"name": "x"}], columns=2)

        with self.assertRaises(NotImplementedError):
            nb.show_metrics_table([{"name": "x"}])

        with self.assertRaises(NotImplementedError):
            nb.export_notebook_artifacts({"intermediate": {}}, output_dir="outputs", prefix="demo")

    def test_main_signatures(self) -> None:
        sig_overlay = inspect.signature(nb.make_overlay)
        self.assertIn("image", sig_overlay.parameters)
        self.assertIn("mask", sig_overlay.parameters)
        self.assertIn("alpha", sig_overlay.parameters)

        sig_run = inspect.signature(nb.run_once)
        self.assertIn("run_cfg", sig_run.parameters)
        self.assertIn("pipeline_params", sig_run.parameters)


if __name__ == "__main__":
    unittest.main()
