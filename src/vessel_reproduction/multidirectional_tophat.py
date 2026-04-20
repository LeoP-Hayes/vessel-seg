from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import cv2
import numpy as np


BorderMode = Literal["reflect", "replicate", "constant"]
AnchorMode = Literal["center"]
FuseMethod = Literal["pixel_max"]


@dataclass(frozen=True)
class TopHatOptions:
    num_directions: int = 9
    line_length: int = 6
    border_mode: BorderMode = "reflect"
    eps: float = 1e-6


@dataclass(frozen=True)
class DirectionSpec:
    index: int
    angle_deg: float


@dataclass(frozen=True)
class TopHatIntermediate:
    angles_deg: tuple[float, ...]
    fused_response: np.ndarray
    selems: tuple[np.ndarray, ...] | None = None
    directional_responses: tuple[np.ndarray, ...] | None = None
    directional_backgrounds: tuple[np.ndarray, ...] | None = None


class TopHatConfigError(ValueError):
    """配置参数不合法。"""


class TopHatInputError(ValueError):
    """输入数据不满足接口约束。"""


def validate_tophat_params(*, num_directions: int, line_length: int) -> int:
    if num_directions < 2:
        raise TopHatConfigError("num_directions must be >= 2")
    if line_length < 2:
        raise TopHatConfigError("line_length must be >= 2")
    if line_length % 2 == 0:
        return line_length + 1
    return line_length


def generate_direction_specs(
    *,
    num_directions: int,
    start_deg: float = 0.0,
    end_deg: float = 180.0,
    include_endpoint: bool = True,
) -> list[DirectionSpec]:
    validate_tophat_params(num_directions=num_directions, line_length=3)
    if end_deg <= start_deg:
        raise TopHatConfigError("end_deg must be greater than start_deg")

    angles = np.linspace(start_deg, end_deg, num=num_directions, endpoint=include_endpoint, dtype=np.float64)
    return [DirectionSpec(index=i, angle_deg=float(a)) for i, a in enumerate(angles)]


def generate_line_structuring_element(
    *,
    line_length: int,
    angle_deg: float,
    thickness: int = 1,
    anchor_mode: AnchorMode = "center",
) -> np.ndarray:
    line_length_eff = validate_tophat_params(num_directions=2, line_length=line_length)
    if thickness != 1:
        raise TopHatConfigError("thickness currently only supports 1")
    if anchor_mode != "center":
        raise TopHatConfigError("anchor_mode currently only supports 'center'")

    radius = line_length_eff // 2
    size = line_length_eff
    center = radius

    theta = np.deg2rad(float(angle_deg))
    dx = int(round(radius * np.cos(theta)))
    dy = int(round(radius * np.sin(theta)))

    x0, y0 = center - dx, center - dy
    x1, y1 = center + dx, center + dy

    kernel = np.zeros((size, size), dtype=np.uint8)
    for x, y in _bresenham_line(x0, y0, x1, y1):
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1
    kernel[center, center] = 1
    return kernel


def compute_modified_tophat_response(
    image: np.ndarray,
    *,
    selem: np.ndarray,
    border_mode: BorderMode = "reflect",
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """改进 top-hat：resp = Ieq - min(open(close(Ieq,S),S), Ieq)。"""
    _validate_image_shape(image)
    _validate_selem(selem)
    if eps <= 0:
        raise TopHatConfigError("eps must be > 0")

    border = _to_cv2_border(border_mode)
    img = image.astype(np.float32)

    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, selem, borderType=border)
    smoothed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, selem, borderType=border)

    background = np.minimum(smoothed, img).astype(np.float32)
    response = (img - background).astype(np.float32)
    response = np.clip(np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    return response, background


def fuse_directional_responses(
    responses: Sequence[np.ndarray],
    *,
    method: FuseMethod = "pixel_max",
) -> np.ndarray:
    if len(responses) == 0:
        raise TopHatInputError("responses must be non-empty")
    if method != "pixel_max":
        raise TopHatConfigError("only 'pixel_max' is supported")

    first_shape = responses[0].shape
    for idx, arr in enumerate(responses):
        _validate_image_shape(arr)
        if arr.shape != first_shape:
            raise TopHatInputError(f"response shape mismatch at index {idx}")

    stacked = np.stack(responses, axis=0).astype(np.float32)
    return np.max(stacked, axis=0).astype(np.float32)


def run_multidirectional_tophat(
    image: np.ndarray,
    *,
    num_directions: int,
    line_length: int,
    save_intermediate: bool = False,
    save_directional: bool = False,
    border_mode: BorderMode = "reflect",
) -> tuple[np.ndarray, TopHatIntermediate | None]:
    _validate_image_shape(image)
    line_length_eff = validate_tophat_params(num_directions=num_directions, line_length=line_length)
    if save_directional and not save_intermediate:
        raise TopHatConfigError("save_directional requires save_intermediate=True")

    specs = generate_direction_specs(num_directions=num_directions, start_deg=0.0, end_deg=180.0, include_endpoint=True)

    resp_list: list[np.ndarray] = []
    bg_list: list[np.ndarray] = []
    se_list: list[np.ndarray] = []

    for spec in specs:
        se = generate_line_structuring_element(line_length=line_length_eff, angle_deg=spec.angle_deg)
        resp, bg = compute_modified_tophat_response(image, selem=se, border_mode=border_mode)
        resp_list.append(resp)
        bg_list.append(bg)
        se_list.append(se)

    fused = fuse_directional_responses(resp_list, method="pixel_max")

    if not save_intermediate:
        return fused, None

    inter = TopHatIntermediate(
        angles_deg=tuple(s.angle_deg for s in specs),
        fused_response=fused,
        selems=tuple(se_list) if save_directional else None,
        directional_responses=tuple(resp_list) if save_directional else None,
        directional_backgrounds=tuple(bg_list) if save_directional else None,
    )
    return fused, inter


def serialize_tophat_intermediate(
    intermediate: TopHatIntermediate,
    *,
    output_dir: str | Path,
    prefix: str,
    save_directional: bool = False,
) -> list[str]:
    if not prefix:
        raise TopHatConfigError("prefix must be non-empty")
    if not isinstance(intermediate, TopHatIntermediate):
        raise TopHatInputError("intermediate must be TopHatIntermediate")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    fused_path = out_dir / f"{prefix}_tophat_fused.png"
    cv2.imwrite(str(fused_path), _to_uint8(intermediate.fused_response))
    saved.append(str(fused_path))

    if save_directional and intermediate.directional_responses is not None:
        for idx, arr in enumerate(intermediate.directional_responses):
            p = out_dir / f"{prefix}_tophat_dir_{idx:02d}.png"
            cv2.imwrite(str(p), _to_uint8(arr))
            saved.append(str(p))

    if save_directional and intermediate.directional_backgrounds is not None:
        for idx, arr in enumerate(intermediate.directional_backgrounds):
            p = out_dir / f"{prefix}_tophat_bg_{idx:02d}.png"
            cv2.imwrite(str(p), _to_uint8(arr))
            saved.append(str(p))

    return saved


def intermediate_to_rows(intermediate: TopHatIntermediate) -> list[Mapping[str, Any]]:
    if not isinstance(intermediate, TopHatIntermediate):
        raise TopHatInputError("intermediate must be TopHatIntermediate")

    rows: list[Mapping[str, Any]] = [
        {
            "kind": "fused",
            "angle_deg": None,
            "min": float(np.min(intermediate.fused_response)),
            "max": float(np.max(intermediate.fused_response)),
            "mean": float(np.mean(intermediate.fused_response)),
        }
    ]
    if intermediate.directional_responses is not None:
        for idx, arr in enumerate(intermediate.directional_responses):
            rows.append(
                {
                    "kind": "directional",
                    "angle_deg": float(intermediate.angles_deg[idx]),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                }
            )
    return rows


def _validate_image_shape(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TopHatInputError("image must be numpy.ndarray")
    if image.ndim != 2:
        raise TopHatInputError("image must be 2D grayscale array with shape (H, W)")
    if image.size == 0:
        raise TopHatInputError("image must be non-empty")


def _validate_selem(selem: np.ndarray) -> None:
    if not isinstance(selem, np.ndarray):
        raise TopHatInputError("selem must be numpy.ndarray")
    if selem.ndim != 2:
        raise TopHatInputError("selem must be 2D array")
    if selem.size == 0:
        raise TopHatInputError("selem must be non-empty")


def _to_cv2_border(mode: BorderMode) -> int:
    if mode == "reflect":
        return cv2.BORDER_REFLECT_101
    if mode == "replicate":
        return cv2.BORDER_REPLICATE
    if mode == "constant":
        return cv2.BORDER_CONSTANT
    raise TopHatConfigError("invalid border_mode")


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    min_v = float(np.min(a))
    max_v = float(np.max(a))
    if max_v > min_v:
        a = (a - min_v) / (max_v - min_v)
    else:
        a = np.zeros_like(a, dtype=np.float32)
    return np.clip(a * 255.0, 0.0, 255.0).astype(np.uint8)


def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

    return points
