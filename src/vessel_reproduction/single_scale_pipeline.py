from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np

from .illumination_correction import IlluminationCorrectionOptions, illumination_correction
from .multidirectional_tophat import TopHatIntermediate, TopHatOptions, run_multidirectional_tophat


ThresholdMethod = Literal["adaptive_mean"]
MaskDType = Literal["uint8", "bool"]
Connectivity = Literal[4, 8]


@dataclass(frozen=True)
class AdaptiveThresholdOptions:
    method: ThresholdMethod = "adaptive_mean"
    block_size: int = 21
    C: float = -2.0
    input_scale: Literal["float01", "uint8"] = "float01"
    output_dtype: MaskDType = "uint8"


@dataclass(frozen=True)
class AreaDenoiseOptions:
    area_min: int = 30
    connectivity: Connectivity = 8
    output_dtype: MaskDType = "uint8"


@dataclass(frozen=True)
class PipelineDebugOptions:
    return_intermediate: bool = True
    include_directional: bool = False


@dataclass(frozen=True)
class SingleScalePipelineParams:
    illumination: IlluminationCorrectionOptions = field(
        default_factory=lambda: IlluminationCorrectionOptions(window_size=7, c=0.01)
    )
    tophat: TopHatOptions = field(default_factory=TopHatOptions)
    threshold: AdaptiveThresholdOptions = field(default_factory=AdaptiveThresholdOptions)
    denoise: AreaDenoiseOptions = field(default_factory=AreaDenoiseOptions)
    debug: PipelineDebugOptions = field(default_factory=PipelineDebugOptions)


class PipelineConfigError(ValueError):
    """单尺度流水线参数错误。"""


class PipelineInputError(ValueError):
    """单尺度流水线输入不合法。"""


def validate_threshold_options(options: AdaptiveThresholdOptions) -> None:
    if options.method != "adaptive_mean":
        raise PipelineConfigError("only adaptive_mean is supported")
    if options.block_size < 3 or options.block_size % 2 == 0:
        raise PipelineConfigError("threshold.block_size must be odd and >= 3")


def validate_denoise_options(options: AreaDenoiseOptions) -> None:
    if options.area_min < 0:
        raise PipelineConfigError("denoise.area_min must be >= 0")
    if options.connectivity not in (4, 8):
        raise PipelineConfigError("denoise.connectivity must be 4 or 8")


def validate_pipeline_input(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise PipelineInputError("image must be numpy.ndarray")
    if image.ndim != 2:
        raise PipelineInputError("image must be 2D grayscale array with shape (H, W)")
    if image.size == 0:
        raise PipelineInputError("image must be non-empty")


def adaptive_threshold_single(
    response: np.ndarray,
    *,
    options: AdaptiveThresholdOptions,
) -> np.ndarray:
    """自适应阈值。"""
    validate_pipeline_input(response)
    validate_threshold_options(options)

    if options.input_scale == "float01":
        src = np.clip(response.astype(np.float32), 0.0, 1.0)
        src_u8 = (src * 255.0).astype(np.uint8)
    elif options.input_scale == "uint8":
        if response.dtype == np.uint8:
            src_u8 = response
        else:
            src_u8 = np.clip(response, 0, 255).astype(np.uint8)
    else:
        raise PipelineConfigError("invalid threshold.input_scale")

    binary = cv2.adaptiveThreshold(
        src_u8,
        maxValue=1,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=options.block_size,
        C=float(options.C),
    )

    if options.output_dtype == "bool":
        return (binary > 0)
    if options.output_dtype == "uint8":
        return (binary > 0).astype(np.uint8)
    raise PipelineConfigError("invalid threshold.output_dtype")


def remove_small_components(
    mask: np.ndarray,
    *,
    options: AreaDenoiseOptions,
) -> np.ndarray:
    """面积去噪：移除面积小于 area_min 的连通域。"""
    validate_pipeline_input(mask)
    validate_denoise_options(options)

    bin_u8 = (mask > 0).astype(np.uint8)
    if options.area_min == 0:
        cleaned = bin_u8
    else:
        num, labels, stats, _cent = cv2.connectedComponentsWithStats(bin_u8, connectivity=int(options.connectivity))
        cleaned = np.zeros_like(bin_u8, dtype=np.uint8)
        for lab in range(1, num):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area >= options.area_min:
                cleaned[labels == lab] = 1

    if options.output_dtype == "bool":
        return cleaned.astype(bool)
    if options.output_dtype == "uint8":
        return cleaned.astype(np.uint8)
    raise PipelineConfigError("invalid denoise.output_dtype")


def run_single_scale_pipeline(
    image: np.ndarray,
    params: SingleScalePipelineParams,
) -> tuple[np.ndarray, dict[str, Any]]:
    """单尺度分割流水线主函数。"""
    validate_pipeline_input(image)
    validate_threshold_options(params.threshold)
    validate_denoise_options(params.denoise)

    intermediate: dict[str, Any] = {
        "input": image,
        "illumination_corrected": None,
        "tophat_fused": None,
        "tophat_directional": None,
        "threshold_binary": None,
        "denoised_binary": None,
        "meta": {
            "shape": image.shape,
            "dtype_input": str(image.dtype),
            "dtype_output": str(params.denoise.output_dtype),
            "threshold_method": params.threshold.method,
            "connectivity": params.denoise.connectivity,
        },
    }

    ieq = illumination_correction(
        image,
        window_size=params.illumination.window_size,
        c=params.illumination.c,
        in_range=params.illumination.in_range,
        out_dtype=params.illumination.out_dtype,
        out_range=params.illumination.out_range,
        border_mode=params.illumination.border_mode,
        eps=params.illumination.eps,
    )
    intermediate["illumination_corrected"] = ieq

    tophat_fused, tophat_intermediate = run_multidirectional_tophat(
        ieq,
        num_directions=params.tophat.num_directions,
        line_length=params.tophat.line_length,
        save_intermediate=params.debug.include_directional,
        save_directional=params.debug.include_directional,
        border_mode=params.tophat.border_mode,
    )
    intermediate["tophat_fused"] = tophat_fused
    if isinstance(tophat_intermediate, TopHatIntermediate):
        intermediate["tophat_directional"] = tophat_intermediate.directional_responses

    mask_thr = adaptive_threshold_single(tophat_fused, options=params.threshold)
    intermediate["threshold_binary"] = mask_thr

    mask_final = remove_small_components(mask_thr, options=params.denoise)
    intermediate["denoised_binary"] = mask_final

    if not params.debug.return_intermediate:
        return mask_final, {"meta": intermediate["meta"]}
    return mask_final, intermediate
