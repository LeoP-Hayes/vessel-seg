from __future__ import annotations

import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "run_infer.py"

spec = importlib.util.spec_from_file_location("run_infer_script", SCRIPT_PATH)
assert spec and spec.loader
run_infer = importlib.util.module_from_spec(spec)
sys.modules["run_infer_script"] = run_infer
spec.loader.exec_module(run_infer)


class TestRunInferInterfaces(unittest.TestCase):
    def test_parse_args_defaults(self) -> None:
        args = run_infer.parse_args([])
        self.assertEqual(args.config, "config/default.yaml")
        self.assertEqual(args.pipeline, "dual")
        self.assertEqual(args.input_mode, "auto")

    def test_parse_args_flags(self) -> None:
        args = run_infer.parse_args(
            [
                "--config",
                "config/default.yaml",
                "--input_dir",
                "data",
                "--output_dir",
                "outputs",
                "--pipeline",
                "single",
                "--input_mode",
                "split_dirs",
                "--recursive",
                "--save_intermediate",
                "--save_directional",
                "--skip_on_error",
                "--max_errors",
                "7",
                "--log_level",
                "DEBUG",
                "--dry_run",
            ]
        )
        self.assertEqual(args.pipeline, "single")
        self.assertTrue(args.recursive)
        self.assertTrue(args.skip_on_error)
        self.assertEqual(args.max_errors, 7)
        self.assertTrue(args.dry_run)

    def test_overrides_mapping(self) -> None:
        args = run_infer.parse_args(["--input_dir", "data", "--output_dir", "outputs", "--save_intermediate", "--max_errors", "8"])
        overrides = run_infer.to_overrides(args)
        self.assertEqual(overrides["engineering.io.input_dir"], "data")
        self.assertEqual(overrides["engineering.io.output_dir"], "outputs")
        self.assertTrue(overrides["engineering.output.save_intermediate"])
        self.assertEqual(overrides["engineering.robustness.max_errors"], 8)

    def test_output_layout_structure(self) -> None:
        layout = run_infer.build_output_layout(Path("outputs"), "run_config_snapshot.yaml", "metrics_unsup.csv")
        self.assertEqual(layout.masks_dir, Path("outputs") / "masks")
        self.assertEqual(layout.review_csv_path, Path("outputs") / "review_list.csv")

    def test_main_signature(self) -> None:
        sig = inspect.signature(run_infer.main)
        self.assertIn("argv", sig.parameters)


if __name__ == "__main__":
    unittest.main()
