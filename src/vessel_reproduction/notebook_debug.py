from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np


ImageRange = Literal["auto", "uint8", "float01"]
MaskMode = Literal["binary01", "binary255", "bool"]


NOTEBOOK_SECTIONS: tuple[str, ...] = (
    "00_setup",
    "01_load_one_sample",
    "02_run_single_scale_debug",
    "03_run_dual_scale_debug",
    "04_visualization_panels",
    "05_parameter_sweep_compare",
    "06_export_and_review",
)


DEFAULT_PANELS: tuple[str, ...] = (
    "input",
    "illumination_corrected",
    "tophat_fused",
    "threshold_binary",
    "denoised_binary",
    "overlay",
)


@dataclass(frozen=True)
class NotebookRunConfig:
    sample_path: str
    use_dual_scale: bool = True
    save_outputs: bool = False


class NotebookConfigError(ValueError):
    """Notebook 调试配置不合法。"""


class NotebookInputError(ValueError):
    """Notebook 调试输入不合法。"""


def make_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    image_range: ImageRange = "auto",
    mask_mode: MaskMode = "binary01",
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
    as_rgb: bool = True,
) -> np.ndarray:
    """生成单图 overlay。

    输入：
    - `image`: (H, W) 灰度图
    - `mask`: (H, W) 二值图

    输出：
    - `overlay`: (H, W, 3) uint8
    """
    _validate_image_mask_pair(image, mask)
    if not (0.0 <= alpha <= 1.0):
        raise NotebookConfigError("alpha must be in [0, 1]")
    if len(color) != 3:
        raise NotebookConfigError("color must be RGB tuple with 3 elements")

    base = _normalize_image_to_uint8(image, image_range=image_range)
    base_rgb = np.stack([base, base, base], axis=-1)

    binary = _normalize_mask_to_bool(mask, mask_mode=mask_mode)
    out = base_rgb.astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)

    out[binary] = (1.0 - alpha) * out[binary] + alpha * color_arr
    out = np.clip(out, 0, 255).astype(np.uint8)

    if not as_rgb:
        # 保留接口位，当前返回 RGB。
        pass
    return out


def run_once(
    run_cfg: NotebookRunConfig,
    pipeline_params: Any,
) -> dict[str, Any]:
    """Notebook 一键重跑接口。

    预期（实现阶段）：
    1. 读取单图
    2. 调用 single/dual scale pipeline
    3. 生成 overlay
    4. 返回可视化结果包
    """
    if not run_cfg.sample_path:
        raise NotebookConfigError("sample_path must be non-empty")
    _ = pipeline_params
    raise NotImplementedError("Interface only; implementation intentionally omitted.")


def compare_param_sets(
    image: np.ndarray,
    param_sets: Sequence[tuple[str, Any]],
) -> list[dict[str, Any]]:
    """同图多参数组对比接口。"""
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise NotebookInputError("image must be 2D grayscale numpy.ndarray")
    if len(param_sets) == 0:
        return []
    for name, _ in param_sets:
        if not name:
            raise NotebookConfigError("param set name must be non-empty")
    raise NotImplementedError("Interface only; implementation intentionally omitted.")


def show_debug_panels(result: Mapping[str, Any], *, show_directional: bool = False) -> None:
    """中间结果面板展示接口（Notebook可视化占位）。"""
    _ = show_directional
    if "intermediate" not in result:
        raise NotebookInputError("result must contain 'intermediate'")
    raise NotImplementedError("Interface only; implementation intentionally omitted.")


def show_comparison_grid(results: Sequence[Mapping[str, Any]], *, columns: int = 3) -> None:
    """参数对比网格展示接口（Notebook可视化占位）。"""
    if columns < 1:
        raise NotebookConfigError("columns must be >= 1")
    if len(results) == 0:
        return None
    raise NotImplementedError("Interface only; implementation intentionally omitted.")


def show_metrics_table(results: Sequence[Mapping[str, Any]]) -> Any:
    """结果指标表接口。

    预期（实现阶段）：返回 pandas.DataFrame。
    """
    if len(results) == 0:
        return []
    raise NotImplementedError("Interface only; implementation intentionally omitted.")


def export_notebook_artifacts(
    result: Mapping[str, Any],
    *,
    output_dir: str | Path,
    prefix: str,
) -> list[str]:
    """Notebook 调试产物导出接口（mask/overlay/intermediate）。"""
    if not prefix:
        raise NotebookConfigError("prefix must be non-empty")
    if "intermediate" not in result:
        raise NotebookInputError("result must contain 'intermediate'")
    _ = Path(output_dir)
    raise NotImplementedError("Interface only; implementation intentionally omitted.")


def _validate_image_mask_pair(image: np.ndarray, mask: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise NotebookInputError("image must be 2D grayscale numpy.ndarray")
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise NotebookInputError("mask must be 2D binary numpy.ndarray")
    if image.shape != mask.shape:
        raise NotebookInputError("image and mask must have the same shape")


def _normalize_image_to_uint8(image: np.ndarray, *, image_range: ImageRange) -> np.ndarray:
    if image_range == "uint8":
        if image.dtype != np.uint8:
            raise NotebookInputError("image_range='uint8' expects uint8 image")
        return image

    image_f = image.astype(np.float32)
    if image_range == "float01":
        image_f = np.clip(image_f, 0.0, 1.0) * 255.0
        return image_f.astype(np.uint8)

    # auto
    if image.dtype == np.uint8:
        return image
    max_val = float(np.max(image_f)) if image_f.size > 0 else 0.0
    if max_val <= 1.0:
        image_f = np.clip(image_f, 0.0, 1.0) * 255.0
    else:
        image_f = np.clip(image_f, 0.0, 255.0)
    return image_f.astype(np.uint8)


def _normalize_mask_to_bool(mask: np.ndarray, *, mask_mode: MaskMode) -> np.ndarray:
    if mask_mode == "bool":
        return mask.astype(bool)
    if mask_mode == "binary01":
        return mask.astype(np.float32) > 0.5
    if mask_mode == "binary255":
        return mask.astype(np.float32) > 127.0
    raise NotebookConfigError("invalid mask_mode")
