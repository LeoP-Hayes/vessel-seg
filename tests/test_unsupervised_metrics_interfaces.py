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
um = importlib.import_module("vessel_reproduction.unsupervised_metrics")


class TestUnsupervisedMetricsInterfaces(unittest.TestCase):
    def test_required_symbols_are_exported(self) -> None:
        names = {
            "UnsupervisedMetrics",
            "MetricsComputeOptions",
            "OutlierRuleConfig",
            "MetricThreshold",
            "AnomalyDecision",
            "METRICS_UNSUP_FIELDS",
            "REVIEW_LIST_FIELDS",
            "MetricsConfigError",
            "MetricsInputError",
            "validate_metrics_mask",
            "validate_outlier_rule_config",
            "compute_unsupervised_metrics",
            "compute_metrics_for_batch",
            "fit_outlier_thresholds",
            "mark_anomalies",
            "export_metrics_csv",
            "export_review_csv",
            "metrics_row_template",
        }
        for name in names:
            self.assertTrue(hasattr(vr, name), f"missing export: {name}")

    def test_compute_and_mark(self) -> None:
        m1 = np.zeros((32, 32), dtype=np.uint8)
        m1[10:20, 10:20] = 1
        m2 = np.zeros((32, 32), dtype=np.uint8)
        m2[5:27, 15:17] = 1

        rows = []
        for i, m in enumerate([m1, m2], start=1):
            met = um.compute_unsupervised_metrics(m)
            rows.append(
                {
                    "sample_id": f"s{i}",
                    "split": "test",
                    "rel_path": f"{i}.png",
                    "mask_path": f"m{i}.png",
                    "height": 32,
                    "width": 32,
                    "vessel_area_ratio": met.vessel_area_ratio,
                    "connected_components_count": met.connected_components_count,
                    "largest_component_ratio": met.largest_component_ratio,
                    "skeleton_length_px": met.skeleton_length_px,
                    "mean_branch_degree": met.mean_branch_degree,
                    "branch_points_count": met.branch_points_count,
                    "endpoints_count": met.endpoints_count,
                    "metrics_version": "v1",
                    "created_at_utc": "x",
                }
            )

        thresholds = um.fit_outlier_thresholds(rows)
        annotated, review = um.mark_anomalies(rows, thresholds)
        self.assertEqual(len(annotated), 2)
        self.assertIsInstance(review, list)

    def test_export_csv(self) -> None:
        row = {
            "sample_id": "s1",
            "split": "test",
            "rel_path": "1.png",
            "mask_path": "m1.png",
            "height": 32,
            "width": 32,
            "vessel_area_ratio": 0.1,
            "connected_components_count": 1,
            "largest_component_ratio": 1.0,
            "skeleton_length_px": 10.0,
            "mean_branch_degree": 0.0,
            "branch_points_count": 0,
            "endpoints_count": 2,
            "metrics_version": "v1",
            "created_at_utc": "x",
        }
        with tempfile.TemporaryDirectory() as td:
            p1 = um.export_metrics_csv([row], Path(td) / "metrics_unsup.csv")
            p2 = um.export_review_csv([], Path(td) / "review_list.csv")
            self.assertTrue(Path(p1).exists())
            self.assertTrue(Path(p2).exists())

    def test_main_signature(self) -> None:
        sig = inspect.signature(um.compute_unsupervised_metrics)
        self.assertIn("mask", sig.parameters)


if __name__ == "__main__":
    unittest.main()
