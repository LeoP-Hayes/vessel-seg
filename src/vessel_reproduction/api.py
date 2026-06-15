from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from .config_loader import load_config
from .config_schema import AppConfig
from .data_manifest import DiscoverOptions, discover_samples, read_grayscale_image
from .dual_scale_pipeline import DualScalePipelineParams, run_dual_scale_pipeline, run_triple_scale_pipeline
from .single_scale_pipeline import PipelineDebugOptions, SingleScalePipelineParams, run_single_scale_pipeline


PipelineName = Literal["single", "dual", "triple"]
SplitMode = Literal["train_val_test", "single_dir"]
OnError = Literal["raise", "skip"]


@dataclass(frozen=True)
class BatchExtractionSummary:
    total: int
    processed: int
    skipped: int
    failed: int
    failed_samples: tuple[str, ...]


def extract_vessel_mask(
    image_or_path: np.ndarray | str | Path,
    *,
    pipeline: PipelineName = "dual",
    config_path: str | Path | None = "config/default.yaml",
    config: AppConfig | None = None,
    return_intermediate: bool = False,
    include_directional: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """提取单图血管掩码。

    返回 `uint8` 二值图，取值为 `0/1`。
    """
    app_cfg = _resolve_config(config=config, config_path=config_path)
    image = _load_input_image(image_or_path)
    image = _apply_preprocess(image, invert_intensity=bool(app_cfg.algorithm.preprocess.invert_intensity))

    if pipeline == "single":
        params = _build_single_scale_params(app_cfg, return_intermediate=return_intermediate, include_directional=include_directional)
        mask, intermediate = run_single_scale_pipeline(image, params)
    elif pipeline == "dual":
        params = _build_dual_scale_params(app_cfg, return_intermediate=return_intermediate, include_directional=include_directional)
        mask, intermediate = run_dual_scale_pipeline(image, params)
    elif pipeline == "triple":
        params = _build_triple_scale_params(app_cfg, return_intermediate=return_intermediate, include_directional=include_directional)
        mask, intermediate = run_triple_scale_pipeline(image, params)
    else:
        raise ValueError(f"unsupported pipeline: {pipeline}")

    mask_u8 = (mask > 0).astype(np.uint8)
    if return_intermediate:
        return mask_u8, intermediate
    return mask_u8


def extract_vessel_masks_batch(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    split_mode: SplitMode = "train_val_test",
    image_subdir: str = "B",
    output_subdir: str = "B_mask",
    pipeline: PipelineName = "dual",
    config_path: str | Path | None = "config/default.yaml",
    config: AppConfig | None = None,
    recursive: bool = False,
    overwrite: bool = False,
    on_error: OnError = "raise",
) -> BatchExtractionSummary:
    """批量提取血管掩码并写入输出目录。"""
    app_cfg = _resolve_config(config=config, config_path=config_path)
    input_root = Path(input_dir)
    output_root = Path(output_dir)

    total = 0
    processed = 0
    skipped = 0
    failed = 0
    failed_samples: list[str] = []

    if split_mode == "train_val_test":
        split_specs = (
            ("train", input_root / "train" / image_subdir, output_root / "train" / output_subdir),
            ("val", input_root / "val" / image_subdir, output_root / "val" / output_subdir),
            ("test", input_root / "test" / image_subdir, output_root / "test" / output_subdir),
        )
    elif split_mode == "single_dir":
        split_specs = (("all", input_root, output_root),)
    else:
        raise ValueError(f"unsupported split_mode: {split_mode}")

    for split_name, src_dir, dst_dir in split_specs:
        if not src_dir.exists():
            if split_mode == "train_val_test":
                raise FileNotFoundError(f"missing split directory: {src_dir}")
            continue

        samples = discover_samples(src_dir, options=DiscoverOptions(input_mode="single_dir", recursive=recursive))
        total += len(samples)
        for sample in samples:
            dst_path = (dst_dir / sample.rel_path).with_suffix(".png")
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            if dst_path.exists() and not overwrite:
                skipped += 1
                continue

            try:
                mask = extract_vessel_mask(
                    sample.abs_path,
                    pipeline=pipeline,
                    config=app_cfg,
                    config_path=None,
                    return_intermediate=False,
                )
                if isinstance(mask, tuple):
                    raise RuntimeError("unexpected tuple return from extract_vessel_mask")
                ok = cv2.imwrite(str(dst_path), mask.astype(np.uint8) * 255)
                if not ok:
                    raise RuntimeError(f"failed to write output image: {dst_path}")
                processed += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                failed_samples.append(f"{split_name}:{sample.rel_path}:{type(exc).__name__}:{exc}")
                if on_error == "raise":
                    raise

    return BatchExtractionSummary(
        total=total,
        processed=processed,
        skipped=skipped,
        failed=failed,
        failed_samples=tuple(failed_samples),
    )


def _resolve_config(*, config: AppConfig | None, config_path: str | Path | None) -> AppConfig:
    if config is not None:
        return config
    if config_path is None:
        return AppConfig()

    p = Path(config_path)
    if p.exists():
        return load_config(p)

    # 默认路径在外部工程调用时可能不存在，回退到 dataclass 默认值。
    if str(config_path) == "config/default.yaml":
        return AppConfig()
    raise FileNotFoundError(f"config not found: {config_path}")


def _load_input_image(image_or_path: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image_or_path, np.ndarray):
        image = image_or_path
    else:
        image, _meta = read_grayscale_image(image_or_path)

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be numpy.ndarray")
    if image.ndim != 2:
        raise ValueError("image must be 2D grayscale")
    return image


def _build_single_scale_params(
    config: AppConfig,
    *,
    return_intermediate: bool,
    include_directional: bool,
) -> SingleScalePipelineParams:
    algo = config.algorithm

    from .illumination_correction import IlluminationCorrectionOptions
    from .multidirectional_tophat import TopHatOptions
    from .single_scale_pipeline import AdaptiveThresholdOptions, AreaDenoiseOptions

    illumination = IlluminationCorrectionOptions(
        window_size=int(algo.illumination.mean_filter_size_per_scale[0]),
        c=float(algo.illumination.epsilon_c),
        in_range="auto",
        out_dtype="float32",
        out_range="clip01",
        border_mode="reflect",
        eps=1e-6,
    )
    tophat = TopHatOptions(
        num_directions=int(algo.morphology.num_directions),
        line_length=int(algo.morphology.line_length_per_scale[0]),
        border_mode="reflect",
        eps=1e-6,
    )
    threshold = AdaptiveThresholdOptions(
        method=str(algo.threshold.method),
        block_size=int(algo.threshold.adaptive_block_size),
        C=float(algo.threshold.adaptive_C),
        input_scale="float01",
        output_dtype="uint8",
    )
    denoise = AreaDenoiseOptions(
        area_min=int(algo.postprocess.area_min),
        connectivity=8,
        output_dtype="uint8",
    )
    debug = PipelineDebugOptions(return_intermediate=return_intermediate, include_directional=include_directional)
    return SingleScalePipelineParams(
        illumination=illumination,
        tophat=tophat,
        threshold=threshold,
        denoise=denoise,
        debug=debug,
    )


def _build_dual_scale_params(
    config: AppConfig,
    *,
    return_intermediate: bool,
    include_directional: bool,
) -> DualScalePipelineParams:
    algo = config.algorithm
    scales = tuple(float(x) for x in algo.pyramid.scales)
    line_lengths = tuple(int(x) for x in algo.morphology.line_length_per_scale)
    mean_sizes = tuple(int(x) for x in algo.illumination.mean_filter_size_per_scale)
    if len(scales) < 2 or len(line_lengths) < 2 or len(mean_sizes) < 2:
        raise ValueError("dual pipeline requires at least two levels in config")

    base = _build_single_scale_params(
        config,
        return_intermediate=return_intermediate,
        include_directional=include_directional,
    )
    return DualScalePipelineParams(
        scales=(scales[0], scales[1]),
        line_length_per_scale=(line_lengths[0], line_lengths[1]),
        mean_filter_size_per_scale=(mean_sizes[0], mean_sizes[1]),
        base_single_scale=base,
        output_mask_dtype="uint8",
    )


def _build_triple_scale_params(
    config: AppConfig,
    *,
    return_intermediate: bool,
    include_directional: bool,
) -> DualScalePipelineParams:
    algo = config.algorithm
    scales = tuple(float(x) for x in algo.pyramid.scales)
    line_lengths = tuple(int(x) for x in algo.morphology.line_length_per_scale)
    mean_sizes = tuple(int(x) for x in algo.illumination.mean_filter_size_per_scale)
    if len(scales) < 3 or len(line_lengths) < 3 or len(mean_sizes) < 3:
        raise ValueError("triple pipeline requires at least three levels in config")

    base = _build_single_scale_params(
        config,
        return_intermediate=return_intermediate,
        include_directional=include_directional,
    )
    return DualScalePipelineParams(
        scales=(scales[0], scales[1], scales[2]),
        line_length_per_scale=(line_lengths[0], line_lengths[1], line_lengths[2]),
        mean_filter_size_per_scale=(mean_sizes[0], mean_sizes[1], mean_sizes[2]),
        base_single_scale=base,
        output_mask_dtype="uint8",
    )


def _apply_preprocess(image: np.ndarray, *, invert_intensity: bool) -> np.ndarray:
    if not invert_intensity:
        return image
    if image.dtype == np.uint8:
        return (255 - image).astype(np.uint8)
    image_f = image.astype(np.float32)
    max_val = float(np.max(image_f)) if image_f.size else 0.0
    if max_val <= 1.0:
        return np.clip(1.0 - image_f, 0.0, 1.0).astype(np.float32)
    return np.clip(255.0 - image_f, 0.0, 255.0).astype(np.float32)
