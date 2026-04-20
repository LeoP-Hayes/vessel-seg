from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


InputRange = Literal["auto", "uint8", "float01"]
OutputDType = Literal["float32", "uint8"]
OutputRange = Literal["none", "clip01", "minmax01"]
BorderMode = Literal["reflect", "replicate", "constant"]


@dataclass(frozen=True)
class IlluminationCorrectionOptions:
    """光照校正接口参数。"""

    window_size: int
    c: float
    in_range: InputRange = "auto"
    out_dtype: OutputDType = "float32"
    out_range: OutputRange = "clip01"
    border_mode: BorderMode = "reflect"
    eps: float = 1e-6


class IlluminationConfigError(ValueError):
    """参数不满足约束时抛出。"""


class IlluminationInputError(ValueError):
    """输入图像不满足接口约束时抛出。"""


def validate_illumination_params(
    window_size: int,
    c: float,
    *,
    eps: float = 1e-6,
) -> None:
    """校验参数约束。"""
    if window_size < 3 or window_size % 2 == 0:
        raise IlluminationConfigError("window_size must be odd and >= 3")
    if c <= 0:
        raise IlluminationConfigError("c must be > 0")
    if eps <= 0:
        raise IlluminationConfigError("eps must be > 0")


def normalize_input_image(
    image: np.ndarray,
    *,
    in_range: InputRange = "auto",
) -> np.ndarray:
    """输入归一化到 float32 [0,1]。"""
    _validate_image_shape(image)
    arr = image.astype(np.float32)

    if in_range == "uint8":
        if image.dtype != np.uint8:
            raise IlluminationInputError("in_range='uint8' expects uint8 image")
        return np.clip(arr / 255.0, 0.0, 1.0)

    if in_range == "float01":
        return np.clip(arr, 0.0, 1.0)

    # auto
    if image.dtype == np.uint8:
        return np.clip(arr / 255.0, 0.0, 1.0)

    max_val = float(np.max(arr)) if arr.size else 0.0
    if max_val <= 1.0:
        return np.clip(arr, 0.0, 1.0)
    return np.clip(arr / 255.0, 0.0, 1.0)


def apply_mean_filter(
    image: np.ndarray,
    *,
    window_size: int,
    border_mode: BorderMode = "reflect",
) -> np.ndarray:
    """均值滤波，输出尺寸与输入一致。"""
    _validate_image_shape(image)
    validate_illumination_params(window_size=window_size, c=1.0, eps=1e-6)
    border = _to_cv2_border(border_mode)
    out = cv2.blur(image.astype(np.float32), (window_size, window_size), borderType=border)
    return out.astype(np.float32)


def illumination_correction(
    image: np.ndarray,
    *,
    window_size: int,
    c: float,
    in_range: InputRange = "auto",
    out_dtype: OutputDType = "float32",
    out_range: OutputRange = "clip01",
    border_mode: BorderMode = "reflect",
    eps: float = 1e-6,
) -> np.ndarray:
    """光照校正：Ieq = Is / (mean_filter(Is, N) + c)。"""
    _validate_image_shape(image)
    validate_illumination_params(window_size=window_size, c=c, eps=eps)

    is_norm = normalize_input_image(image, in_range=in_range)
    mean_img = apply_mean_filter(is_norm, window_size=window_size, border_mode=border_mode)

    denom = np.maximum(mean_img + float(c), float(eps))
    ieq = is_norm / denom
    ieq = np.nan_to_num(ieq, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    if out_range == "clip01":
        ieq = np.clip(ieq, 0.0, 1.0)
    elif out_range == "minmax01":
        min_v = float(np.min(ieq))
        max_v = float(np.max(ieq))
        if max_v > min_v:
            ieq = (ieq - min_v) / (max_v - min_v)
        else:
            ieq = np.zeros_like(ieq, dtype=np.float32)
    elif out_range != "none":
        raise IlluminationConfigError("invalid out_range")

    if out_dtype == "float32":
        return ieq.astype(np.float32)
    if out_dtype == "uint8":
        if out_range == "none":
            ieq = np.clip(ieq, 0.0, 1.0)
        return np.clip(ieq * 255.0, 0.0, 255.0).astype(np.uint8)

    raise IlluminationConfigError("invalid out_dtype")


def _validate_image_shape(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise IlluminationInputError("image must be a numpy.ndarray")
    if image.ndim != 2:
        raise IlluminationInputError("image must be 2D grayscale array with shape (H, W)")
    if image.size == 0:
        raise IlluminationInputError("image must be non-empty")


def _to_cv2_border(mode: BorderMode) -> int:
    if mode == "reflect":
        return cv2.BORDER_REFLECT_101
    if mode == "replicate":
        return cv2.BORDER_REPLICATE
    if mode == "constant":
        return cv2.BORDER_CONSTANT
    raise IlluminationConfigError("invalid border_mode")
