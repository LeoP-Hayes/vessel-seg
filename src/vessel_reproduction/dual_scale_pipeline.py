from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np

from .single_scale_pipeline import SingleScalePipelineParams, run_single_scale_pipeline


InterpDown = Literal["area"]
InterpUpBinary = Literal["nearest"]
MaskDType = Literal["uint8", "bool"]


@dataclass(frozen=True)
class AlignOptions:
    downsample_interpolation: InterpDown = "area"
    upsample_binary_interpolation: InterpUpBinary = "nearest"
    binarize_after_upsample: bool = True
    binary_threshold: float = 0.0


@dataclass(frozen=True)
class MultiScaleDebugOptions:
    return_intermediate: bool = True
    include_scale_details: bool = True


@dataclass(frozen=True)
class DualScalePipelineParams:
    scales: tuple[float, ...] = (1.0, 0.5)
    line_length_per_scale: tuple[int, ...] = (6, 3)
    mean_filter_size_per_scale: tuple[int, ...] = (7, 5)
    base_single_scale: SingleScalePipelineParams = field(default_factory=SingleScalePipelineParams)
    align: AlignOptions = field(default_factory=AlignOptions)
    debug: MultiScaleDebugOptions = field(default_factory=MultiScaleDebugOptions)
    output_mask_dtype: MaskDType = "uint8"


class DualScaleConfigError(ValueError):
    """双尺度流水线参数错误。"""


class DualScaleInputError(ValueError):
    """双尺度流水线输入不合法。"""


def validate_dual_scale_input(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise DualScaleInputError("image must be numpy.ndarray")
    if image.ndim != 2:
        raise DualScaleInputError("image must be 2D grayscale array with shape (H, W)")
    if image.size == 0:
        raise DualScaleInputError("image must be non-empty")


def validate_dual_scale_params(params: DualScalePipelineParams) -> None:
    if len(params.scales) < 2:
        raise DualScaleConfigError("scales must contain at least 2 values")
    if params.scales[0] != 1.0:
        raise DualScaleConfigError("the first scale must be 1.0 for reference size")
    for idx, scale in enumerate(params.scales[1:], start=1):
        if scale <= 0.0 or scale >= 1.0:
            raise DualScaleConfigError(f"scale at index {idx} must be in (0, 1)")
        if scale >= params.scales[idx - 1]:
            raise DualScaleConfigError("scales must be strictly descending after 1.0")

    if len(params.line_length_per_scale) != len(params.scales):
        raise DualScaleConfigError("line_length_per_scale length must match scales")
    if len(params.mean_filter_size_per_scale) != len(params.scales):
        raise DualScaleConfigError("mean_filter_size_per_scale length must match scales")

    for value in params.line_length_per_scale:
        if value < 2:
            raise DualScaleConfigError("line_length_per_scale values must be >= 2")
    for value in params.mean_filter_size_per_scale:
        if value < 3 or value % 2 == 0:
            raise DualScaleConfigError("mean_filter_size_per_scale values must be odd and >= 3")

    if params.align.downsample_interpolation != "area":
        raise DualScaleConfigError("only 'area' downsample interpolation is supported")
    if params.align.upsample_binary_interpolation != "nearest":
        raise DualScaleConfigError("only 'nearest' upsample interpolation is supported")


def build_single_scale_params_for_level(
    base: SingleScalePipelineParams,
    *,
    line_length: int,
    mean_filter_size: int,
) -> SingleScalePipelineParams:
    """为某个尺度构建独立参数快照。"""
    illum = base.illumination
    tophat = base.tophat

    return SingleScalePipelineParams(
        illumination=type(illum)(
            window_size=mean_filter_size,
            c=illum.c,
            in_range=illum.in_range,
            out_dtype=illum.out_dtype,
            out_range=illum.out_range,
            border_mode=illum.border_mode,
            eps=illum.eps,
        ),
        tophat=type(tophat)(
            num_directions=tophat.num_directions,
            line_length=line_length,
            border_mode=tophat.border_mode,
            eps=tophat.eps,
        ),
        threshold=base.threshold,
        denoise=base.denoise,
        debug=base.debug,
    )


def downsample_image(
    image: np.ndarray,
    *,
    scale: float,
    interpolation: InterpDown = "area",
) -> np.ndarray:
    """下采样。"""
    validate_dual_scale_input(image)
    if scale <= 0:
        raise DualScaleConfigError("scale must be > 0")
    if interpolation != "area":
        raise DualScaleConfigError("only 'area' interpolation is supported")

    h, w = image.shape
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    out = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out


def upsample_binary_mask_to_shape(
    mask: np.ndarray,
    *,
    target_shape: tuple[int, int],
    interpolation: InterpUpBinary = "nearest",
    binarize_after_upsample: bool = True,
    binary_threshold: float = 0.0,
    output_dtype: MaskDType = "uint8",
) -> np.ndarray:
    """二值图上采样并对齐到目标尺寸。"""
    validate_dual_scale_input(mask)
    if target_shape[0] <= 0 or target_shape[1] <= 0:
        raise DualScaleConfigError("target_shape must be positive")
    if interpolation != "nearest":
        raise DualScaleConfigError("only 'nearest' interpolation is supported")

    src = (mask > 0).astype(np.uint8)
    resized = cv2.resize(src, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    if binarize_after_upsample:
        resized = (resized > float(binary_threshold)).astype(np.uint8)

    if output_dtype == "bool":
        return resized.astype(bool)
    if output_dtype == "uint8":
        return resized.astype(np.uint8)
    raise DualScaleConfigError("invalid output_dtype")


def fuse_masks_or(mask_s0: np.ndarray, mask_s1_up: np.ndarray, *, output_dtype: MaskDType = "uint8") -> np.ndarray:
    """最终 OR 融合逻辑（逐像素 OR）。"""
    validate_dual_scale_input(mask_s0)
    validate_dual_scale_input(mask_s1_up)
    if mask_s0.shape != mask_s1_up.shape:
        raise DualScaleInputError("mask shapes must match for OR fusion")

    fused = np.logical_or(mask_s0 > 0, mask_s1_up > 0)
    if output_dtype == "bool":
        return fused.astype(bool)
    if output_dtype == "uint8":
        return fused.astype(np.uint8)
    raise DualScaleConfigError("invalid output_dtype")


def fuse_masks_or_many(masks: list[np.ndarray], *, output_dtype: MaskDType = "uint8") -> np.ndarray:
    """多尺度逐像素 OR 融合。"""
    if len(masks) == 0:
        raise DualScaleInputError("masks must be non-empty")

    first_shape = masks[0].shape
    fused = np.zeros(first_shape, dtype=bool)
    for idx, mask in enumerate(masks):
        validate_dual_scale_input(mask)
        if mask.shape != first_shape:
            raise DualScaleInputError(f"mask shape mismatch at index {idx}")
        fused = np.logical_or(fused, mask > 0)

    if output_dtype == "bool":
        return fused.astype(bool)
    if output_dtype == "uint8":
        return fused.astype(np.uint8)
    raise DualScaleConfigError("invalid output_dtype")


def run_dual_scale_pipeline(
    image: np.ndarray,
    params: DualScalePipelineParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    """双尺度推理主入口。"""
    if len(params.scales) != 2:
        raise DualScaleConfigError("run_dual_scale_pipeline expects exactly 2 scales")
    return _run_multi_scale_pipeline(image, params)


def run_triple_scale_pipeline(
    image: np.ndarray,
    params: DualScalePipelineParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    """三尺度推理主入口。"""
    if len(params.scales) != 3:
        raise DualScaleConfigError("run_triple_scale_pipeline expects exactly 3 scales")
    return _run_multi_scale_pipeline(image, params)


def _run_multi_scale_pipeline(
    image: np.ndarray,
    params: DualScalePipelineParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    """多尺度推理核心实现。"""
    validate_dual_scale_input(image)
    validate_dual_scale_params(params)

    h, w = image.shape
    intermediate: dict[str, Any] = {
        "input": image,
        "scale_inputs": {},
        "scale_results": {},
        "aligned_masks": {},
        "fused_or": None,
        "meta": {
            "scales": params.scales,
            "num_scales": len(params.scales),
            "interp_down": params.align.downsample_interpolation,
            "interp_up_binary": params.align.upsample_binary_interpolation,
            "binarize_after_upsample": params.align.binarize_after_upsample,
        },
    }
    aligned_masks: list[np.ndarray] = []
    for idx, scale in enumerate(params.scales):
        key = f"s{idx}"
        if idx == 0:
            image_level = image
        else:
            image_level = downsample_image(
                image,
                scale=scale,
                interpolation=params.align.downsample_interpolation,
            )
        intermediate["scale_inputs"][key] = image_level

        level_params = build_single_scale_params_for_level(
            params.base_single_scale,
            line_length=params.line_length_per_scale[idx],
            mean_filter_size=params.mean_filter_size_per_scale[idx],
        )
        mask_level, inter_level = run_single_scale_pipeline(image_level, level_params)
        intermediate["scale_results"][key] = {"mask": mask_level, "details": inter_level}

        if idx == 0:
            aligned_mask = mask_level
            aligned_key = key
        else:
            aligned_mask = upsample_binary_mask_to_shape(
                mask_level,
                target_shape=(h, w),
                interpolation=params.align.upsample_binary_interpolation,
                binarize_after_upsample=params.align.binarize_after_upsample,
                binary_threshold=params.align.binary_threshold,
                output_dtype=params.output_mask_dtype,
            )
            aligned_key = f"{key}_up"
        intermediate["aligned_masks"][aligned_key] = aligned_mask
        aligned_masks.append(aligned_mask)

    fused = fuse_masks_or_many(aligned_masks, output_dtype=params.output_mask_dtype)
    intermediate["fused_or"] = fused

    if not params.debug.return_intermediate:
        return fused, {"meta": intermediate["meta"]}
    if not params.debug.include_scale_details:
        intermediate["scale_results"] = None
    return fused, intermediate
