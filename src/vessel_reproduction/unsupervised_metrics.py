from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import cv2
import numpy as np


SkeletonMethod = Literal["skimage", "opencv_approx"]
OutlierRuleType = Literal["iqr", "quantile", "mixed"]
Severity = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class UnsupervisedMetrics:
    vessel_area_ratio: float
    connected_components_count: int
    largest_component_ratio: float
    skeleton_length_px: float
    mean_branch_degree: float
    branch_points_count: int
    endpoints_count: int


@dataclass(frozen=True)
class MetricsComputeOptions:
    connectivity: Literal[4, 8] = 8
    skeleton_method: SkeletonMethod = "skimage"


@dataclass(frozen=True)
class OutlierRuleConfig:
    primary_rule: OutlierRuleType = "iqr"
    iqr_k: float = 1.5
    quantile_low: float = 0.01
    quantile_high: float = 0.99
    min_samples_for_iqr: int = 20
    iqr_epsilon: float = 1e-12
    score_threshold: int = 1


@dataclass(frozen=True)
class MetricThreshold:
    metric_name: str
    lower: float
    upper: float
    rule_type: OutlierRuleType


@dataclass(frozen=True)
class AnomalyDecision:
    sample_id: str
    anomaly_score: int
    severity: Severity
    triggered_metrics: tuple[str, ...]
    rule_type: OutlierRuleType


METRICS_UNSUP_FIELDS: tuple[str, ...] = (
    "sample_id",
    "split",
    "rel_path",
    "mask_path",
    "height",
    "width",
    "vessel_area_ratio",
    "connected_components_count",
    "largest_component_ratio",
    "skeleton_length_px",
    "mean_branch_degree",
    "branch_points_count",
    "endpoints_count",
    "metrics_version",
    "created_at_utc",
)


REVIEW_LIST_FIELDS: tuple[str, ...] = (
    "sample_id",
    "split",
    "rel_path",
    "anomaly_score",
    "severity",
    "triggered_metrics",
    "rule_type",
    "threshold_snapshot",
    "key_metric_values",
    "created_at_utc",
)


class MetricsConfigError(ValueError):
    """无监督统计配置错误。"""


class MetricsInputError(ValueError):
    """无监督统计输入错误。"""


METRIC_NAMES: tuple[str, ...] = (
    "vessel_area_ratio",
    "connected_components_count",
    "largest_component_ratio",
    "skeleton_length_px",
    "mean_branch_degree",
    "branch_points_count",
    "endpoints_count",
)


def validate_metrics_mask(mask: np.ndarray) -> None:
    if not isinstance(mask, np.ndarray):
        raise MetricsInputError("mask must be numpy.ndarray")
    if mask.ndim != 2:
        raise MetricsInputError("mask must be 2D binary array with shape (H, W)")
    if mask.size == 0:
        raise MetricsInputError("mask must be non-empty")


def validate_outlier_rule_config(config: OutlierRuleConfig) -> None:
    if config.iqr_k <= 0:
        raise MetricsConfigError("iqr_k must be > 0")
    if not (0 <= config.quantile_low < config.quantile_high <= 1):
        raise MetricsConfigError("quantile bounds must satisfy 0 <= low < high <= 1")
    if config.min_samples_for_iqr < 3:
        raise MetricsConfigError("min_samples_for_iqr must be >= 3")
    if config.iqr_epsilon <= 0:
        raise MetricsConfigError("iqr_epsilon must be > 0")
    if config.score_threshold < 1:
        raise MetricsConfigError("score_threshold must be >= 1")


def compute_unsupervised_metrics(
    mask: np.ndarray,
    *,
    options: MetricsComputeOptions | None = None,
) -> UnsupervisedMetrics:
    validate_metrics_mask(mask)
    opts = options or MetricsComputeOptions()
    if opts.connectivity not in (4, 8):
        raise MetricsConfigError("connectivity must be 4 or 8")
    if opts.skeleton_method not in ("skimage", "opencv_approx"):
        raise MetricsConfigError("invalid skeleton_method")

    bin_mask = (mask > 0).astype(np.uint8)
    h, w = bin_mask.shape
    vessel_pixels = int(np.count_nonzero(bin_mask))

    vessel_area_ratio = float(vessel_pixels / float(h * w))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=int(opts.connectivity))
    comp_count = int(max(0, num - 1))
    if vessel_pixels > 0 and comp_count > 0:
        largest = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
        largest_ratio = float(largest / vessel_pixels)
    else:
        largest_ratio = 0.0

    skeleton = _skeletonize_morph(bin_mask)
    skeleton_u8 = skeleton.astype(np.uint8)
    skeleton_len = float(np.count_nonzero(skeleton_u8))

    neighbor_count = _skeleton_neighbor_count(skeleton_u8)
    branch_mask = (skeleton_u8 > 0) & (neighbor_count >= 3)
    endpoint_mask = (skeleton_u8 > 0) & (neighbor_count == 1)

    branch_points_count = int(np.count_nonzero(branch_mask))
    endpoints_count = int(np.count_nonzero(endpoint_mask))

    if branch_points_count > 0:
        mean_branch_degree = float(np.mean(neighbor_count[branch_mask]))
    else:
        mean_branch_degree = 0.0

    return UnsupervisedMetrics(
        vessel_area_ratio=vessel_area_ratio,
        connected_components_count=comp_count,
        largest_component_ratio=largest_ratio,
        skeleton_length_px=skeleton_len,
        mean_branch_degree=mean_branch_degree,
        branch_points_count=branch_points_count,
        endpoints_count=endpoints_count,
    )


def compute_metrics_for_batch(
    rows: Sequence[Mapping[str, Any]],
    *,
    options: MetricsComputeOptions | None = None,
) -> list[dict[str, Any]]:
    _ = options
    out: list[dict[str, Any]] = []
    for row in rows:
        mask = row.get("mask")
        if mask is None:
            raise MetricsInputError("each row must contain 'mask'")
        metrics = compute_unsupervised_metrics(mask, options=options)
        r = {
            "sample_id": row.get("sample_id", "unknown"),
            "split": row.get("split", "all"),
            "rel_path": row.get("rel_path", ""),
            "mask_path": row.get("mask_path", ""),
            "height": int(mask.shape[0]),
            "width": int(mask.shape[1]),
            "vessel_area_ratio": metrics.vessel_area_ratio,
            "connected_components_count": metrics.connected_components_count,
            "largest_component_ratio": metrics.largest_component_ratio,
            "skeleton_length_px": metrics.skeleton_length_px,
            "mean_branch_degree": metrics.mean_branch_degree,
            "branch_points_count": metrics.branch_points_count,
            "endpoints_count": metrics.endpoints_count,
            "metrics_version": "v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        out.append(r)
    return out


def fit_outlier_thresholds(
    metrics_rows: Sequence[Mapping[str, Any]],
    *,
    config: OutlierRuleConfig | None = None,
) -> dict[str, MetricThreshold]:
    cfg = config or OutlierRuleConfig()
    validate_outlier_rule_config(cfg)
    if len(metrics_rows) == 0:
        return {}

    thresholds: dict[str, MetricThreshold] = {}
    for metric in METRIC_NAMES:
        vals = np.array([float(r[metric]) for r in metrics_rows], dtype=np.float64)

        use_iqr = len(vals) >= cfg.min_samples_for_iqr
        q1 = float(np.quantile(vals, 0.25))
        q3 = float(np.quantile(vals, 0.75))
        iqr = q3 - q1

        if cfg.primary_rule == "iqr" and use_iqr and iqr > cfg.iqr_epsilon:
            low = q1 - cfg.iqr_k * iqr
            high = q3 + cfg.iqr_k * iqr
            rule_type: OutlierRuleType = "iqr"
        else:
            low = float(np.quantile(vals, cfg.quantile_low))
            high = float(np.quantile(vals, cfg.quantile_high))
            rule_type = "quantile"

        thresholds[metric] = MetricThreshold(metric_name=metric, lower=low, upper=high, rule_type=rule_type)

    return thresholds


def mark_anomalies(
    metrics_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, MetricThreshold],
    *,
    config: OutlierRuleConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = config or OutlierRuleConfig()
    validate_outlier_rule_config(cfg)
    if len(metrics_rows) == 0:
        return [], []

    annotated: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    threshold_snapshot = {
        k: {"lower": v.lower, "upper": v.upper, "rule_type": v.rule_type}
        for k, v in thresholds.items()
    }

    for row in metrics_rows:
        triggered: list[str] = []
        rule_types: list[OutlierRuleType] = []

        for metric in METRIC_NAMES:
            if metric not in thresholds:
                continue
            val = float(row[metric])
            th = thresholds[metric]
            if val < th.lower or val > th.upper:
                triggered.append(metric)
                rule_types.append(th.rule_type)

        score = len(triggered)
        if score >= 3:
            severity: Severity = "high"
        elif score == 2:
            severity = "medium"
        else:
            severity = "low"

        if len(set(rule_types)) == 0:
            row_rule: OutlierRuleType = "iqr"
        elif len(set(rule_types)) == 1:
            row_rule = rule_types[0]
        else:
            row_rule = "mixed"

        item = dict(row)
        item["anomaly_score"] = score
        item["severity"] = severity
        item["triggered_metrics"] = "|".join(triggered)
        item["rule_type"] = row_rule
        item["is_anomaly"] = score >= cfg.score_threshold
        annotated.append(item)

        if score >= cfg.score_threshold:
            review_rows.append(
                {
                    "sample_id": row.get("sample_id", "unknown"),
                    "split": row.get("split", "all"),
                    "rel_path": row.get("rel_path", ""),
                    "anomaly_score": score,
                    "severity": severity,
                    "triggered_metrics": "|".join(triggered),
                    "rule_type": row_rule,
                    "threshold_snapshot": json.dumps(threshold_snapshot, ensure_ascii=False),
                    "key_metric_values": json.dumps({m: float(row[m]) for m in METRIC_NAMES}, ensure_ascii=False),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )

    return annotated, review_rows


def export_metrics_csv(metrics_rows: Sequence[Mapping[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(METRICS_UNSUP_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)
    return path


def export_review_csv(review_rows: Sequence[Mapping[str, Any]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(REVIEW_LIST_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in review_rows:
            writer.writerow(row)
    return path


def metrics_row_template(*, sample_id: str, split: str, rel_path: str, mask_path: str, height: int, width: int) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "split": split,
        "rel_path": rel_path,
        "mask_path": mask_path,
        "height": height,
        "width": width,
    }


def _skeletonize_morph(mask01: np.ndarray) -> np.ndarray:
    """OpenCV 近似骨架化（形态学迭代）。"""
    img = (mask01 > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break

    return (skel > 0).astype(np.uint8)


def _skeleton_neighbor_count(skel01: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    cnt = cv2.filter2D(skel01.astype(np.float32), ddepth=cv2.CV_32F, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    return cnt
