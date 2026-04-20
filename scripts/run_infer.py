from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vessel_reproduction.config_loader import export_config_snapshot, load_config
from vessel_reproduction.data_manifest import DiscoverOptions, discover_samples, read_grayscale_image
from vessel_reproduction.dual_scale_pipeline import (
    DualScalePipelineParams,
    run_dual_scale_pipeline,
    run_triple_scale_pipeline,
)
from vessel_reproduction.notebook_debug import make_overlay
from vessel_reproduction.single_scale_pipeline import PipelineDebugOptions, SingleScalePipelineParams, run_single_scale_pipeline
from vessel_reproduction.unsupervised_metrics import (
    MetricsComputeOptions,
    OutlierRuleConfig,
    compute_unsupervised_metrics,
    export_metrics_csv,
    export_review_csv,
    fit_outlier_thresholds,
    mark_anomalies,
)


@dataclass(frozen=True)
class OutputLayout:
    root: Path
    masks_dir: Path
    overlays_dir: Path
    intermediate_dir: Path
    logs_dir: Path
    snapshot_path: Path
    metrics_csv_path: Path
    review_csv_path: Path
    failed_samples_path: Path


@dataclass(frozen=True)
class FailedSample:
    sample_id: str
    rel_path: str
    abs_path: str
    error_type: str
    error_message: str


@dataclass(frozen=True)
class BatchSummary:
    total: int
    ok: int
    failed: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vessel extraction end-to-end CLI")

    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--pipeline", choices=["single", "dual", "triple"], default="dual")
    parser.add_argument("--input_mode", choices=["auto", "single_dir", "split_dirs"], default="auto")
    parser.add_argument("--recursive", action="store_true")

    parser.add_argument("--save_intermediate", action="store_true")
    parser.add_argument("--save_directional", action="store_true")

    parser.add_argument("--skip_on_error", action="store_true")
    parser.add_argument("--max_errors", type=int, default=50)

    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    parser.add_argument("--dry_run", action="store_true")

    return parser.parse_args(argv)


def to_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.input_dir:
        overrides["engineering.io.input_dir"] = args.input_dir
    if args.output_dir:
        overrides["engineering.io.output_dir"] = args.output_dir
    if args.save_intermediate:
        overrides["engineering.output.save_intermediate"] = True
    if args.skip_on_error:
        overrides["engineering.robustness.skip_on_error"] = True
    overrides["engineering.robustness.max_errors"] = args.max_errors
    overrides["engineering.logging.level"] = args.log_level
    return overrides


def build_output_layout(output_root: Path, snapshot_filename: str, metrics_filename: str) -> OutputLayout:
    return OutputLayout(
        root=output_root,
        masks_dir=output_root / "masks",
        overlays_dir=output_root / "overlays",
        intermediate_dir=output_root / "intermediate",
        logs_dir=output_root / "logs",
        snapshot_path=output_root / snapshot_filename,
        metrics_csv_path=output_root / metrics_filename,
        review_csv_path=output_root / "review_list.csv",
        failed_samples_path=output_root / "logs" / "failed_samples.jsonl",
    )


def ensure_output_layout(layout: OutputLayout, *, save_intermediate: bool) -> None:
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.masks_dir.mkdir(parents=True, exist_ok=True)
    layout.overlays_dir.mkdir(parents=True, exist_ok=True)
    layout.logs_dir.mkdir(parents=True, exist_ok=True)
    if save_intermediate:
        layout.intermediate_dir.mkdir(parents=True, exist_ok=True)


def setup_logger(layout: OutputLayout, *, level: str) -> logging.Logger:
    logger = logging.getLogger("run_infer")
    logger.setLevel(getattr(logging, level))
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(layout.logs_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def discover_batch_samples(
    input_dir: str | Path,
    *,
    input_mode: str,
    recursive: bool,
    input_glob: str | None = None,
) -> list[Any]:
    exts = None
    if input_glob:
        exts = _exts_from_glob(input_glob)
    return discover_samples(
        input_dir,
        options=DiscoverOptions(input_mode=input_mode, recursive=recursive, exts=exts or DiscoverOptions().exts),
    )


def build_single_scale_params_from_config(config: Any) -> SingleScalePipelineParams:
    algo = config.algorithm
    eng = config.engineering

    from vessel_reproduction.illumination_correction import IlluminationCorrectionOptions
    from vessel_reproduction.multidirectional_tophat import TopHatOptions
    from vessel_reproduction.single_scale_pipeline import AdaptiveThresholdOptions, AreaDenoiseOptions

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
    debug = PipelineDebugOptions(
        return_intermediate=bool(eng.output.save_intermediate),
        include_directional=False,
    )

    return SingleScalePipelineParams(
        illumination=illumination,
        tophat=tophat,
        threshold=threshold,
        denoise=denoise,
        debug=debug,
    )


def build_dual_scale_params_from_config(config: Any) -> DualScalePipelineParams:
    algo = config.algorithm
    scales = tuple(float(x) for x in algo.pyramid.scales)
    line_lengths = tuple(int(x) for x in algo.morphology.line_length_per_scale)
    mean_sizes = tuple(int(x) for x in algo.illumination.mean_filter_size_per_scale)
    base = build_single_scale_params_from_config(config)
    if len(scales) < 2 or len(line_lengths) < 2 or len(mean_sizes) < 2:
        raise ValueError("dual pipeline requires at least 2 scales/line lengths/mean filter sizes in config")

    return DualScalePipelineParams(
        scales=(scales[0], scales[1]),
        line_length_per_scale=(line_lengths[0], line_lengths[1]),
        mean_filter_size_per_scale=(mean_sizes[0], mean_sizes[1]),
        base_single_scale=base,
        output_mask_dtype="uint8",
    )


def build_triple_scale_params_from_config(config: Any) -> DualScalePipelineParams:
    algo = config.algorithm
    scales = tuple(float(x) for x in algo.pyramid.scales)
    line_lengths = tuple(int(x) for x in algo.morphology.line_length_per_scale)
    mean_sizes = tuple(int(x) for x in algo.illumination.mean_filter_size_per_scale)
    base = build_single_scale_params_from_config(config)
    if len(scales) < 3 or len(line_lengths) < 3 or len(mean_sizes) < 3:
        raise ValueError("triple pipeline requires at least 3 scales/line lengths/mean filter sizes in config")

    return DualScalePipelineParams(
        scales=(scales[0], scales[1], scales[2]),
        line_length_per_scale=(line_lengths[0], line_lengths[1], line_lengths[2]),
        mean_filter_size_per_scale=(mean_sizes[0], mean_sizes[1], mean_sizes[2]),
        base_single_scale=base,
        output_mask_dtype="uint8",
    )


def process_one_sample(sample: Any, *, config: Any, args: argparse.Namespace, layout: OutputLayout) -> dict[str, Any]:
    image, _meta = read_grayscale_image(sample.abs_path)
    image = _apply_preprocess(image, invert_intensity=bool(config.algorithm.preprocess.invert_intensity))

    if args.pipeline == "single":
        params_single = build_single_scale_params_from_config(config)
        params_single = replace(
            params_single,
            debug=replace(
                params_single.debug,
                return_intermediate=bool(config.engineering.output.save_intermediate),
                include_directional=bool(args.save_directional),
            ),
        )
        mask, intermediate = run_single_scale_pipeline(image, params_single)
    elif args.pipeline == "dual":
        params_dual = build_dual_scale_params_from_config(config)
        params_dual = replace(
            params_dual,
            base_single_scale=replace(
                params_dual.base_single_scale,
                debug=replace(
                    params_dual.base_single_scale.debug,
                    return_intermediate=bool(config.engineering.output.save_intermediate),
                    include_directional=bool(args.save_directional),
                ),
            ),
        )
        mask, intermediate = run_dual_scale_pipeline(image, params_dual)
    else:
        params_triple = build_triple_scale_params_from_config(config)
        params_triple = replace(
            params_triple,
            base_single_scale=replace(
                params_triple.base_single_scale,
                debug=replace(
                    params_triple.base_single_scale.debug,
                    return_intermediate=bool(config.engineering.output.save_intermediate),
                    include_directional=bool(args.save_directional),
                ),
            ),
        )
        mask, intermediate = run_triple_scale_pipeline(image, params_triple)

    overlay = make_overlay(image, mask)

    stem = Path(sample.rel_path).with_suffix("").as_posix().replace("/", "__")
    mask_path = layout.masks_dir / f"{stem}_mask.png"
    overlay_path = layout.overlays_dir / f"{stem}_overlay.png"

    if bool(config.engineering.output.save_mask):
        cv2.imwrite(str(mask_path), (mask > 0).astype(np.uint8) * 255)
    if bool(config.engineering.output.save_overlay):
        cv2.imwrite(str(overlay_path), overlay)

    if bool(config.engineering.output.save_intermediate):
        save_intermediate_artifacts(intermediate, layout.intermediate_dir, stem)

    metric = compute_unsupervised_metrics(mask, options=MetricsComputeOptions())
    return {
        "sample_id": sample.sample_id,
        "split": sample.split,
        "rel_path": sample.rel_path,
        "mask_path": str(mask_path),
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "vessel_area_ratio": metric.vessel_area_ratio,
        "connected_components_count": metric.connected_components_count,
        "largest_component_ratio": metric.largest_component_ratio,
        "skeleton_length_px": metric.skeleton_length_px,
        "mean_branch_degree": metric.mean_branch_degree,
        "branch_points_count": metric.branch_points_count,
        "endpoints_count": metric.endpoints_count,
        "metrics_version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_intermediate_artifacts(intermediate: dict[str, Any], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def maybe_save(key: str, arr: Any) -> None:
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            return
        p = out_dir / f"{stem}_{key}.png"
        cv2.imwrite(str(p), _to_uint8(arr))

    maybe_save("illumination", intermediate.get("illumination_corrected"))
    maybe_save("tophat", intermediate.get("tophat_fused"))
    maybe_save("threshold", intermediate.get("threshold_binary"))
    maybe_save("denoised", intermediate.get("denoised_binary"))


def append_failed_sample(path: Path, item: FailedSample) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def run_batch_loop(samples: list[Any], *, config: Any, args: argparse.Namespace, layout: OutputLayout, logger: logging.Logger) -> tuple[list[dict[str, Any]], BatchSummary]:
    metrics_rows: list[dict[str, Any]] = []
    failed = 0

    for idx, sample in enumerate(samples, start=1):
        try:
            row = process_one_sample(sample, config=config, args=args, layout=layout)
            metrics_rows.append(row)
            logger.info("processed %d/%d sample_id=%s", idx, len(samples), sample.sample_id)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.error("failed %d/%d sample_id=%s error=%s", idx, len(samples), getattr(sample, "sample_id", "unknown"), exc)
            append_failed_sample(
                layout.failed_samples_path,
                FailedSample(
                    sample_id=getattr(sample, "sample_id", "unknown"),
                    rel_path=getattr(sample, "rel_path", "unknown"),
                    abs_path=getattr(sample, "abs_path", "unknown"),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                ),
            )
            if not args.skip_on_error:
                raise
            if failed > args.max_errors:
                raise RuntimeError(f"failed samples exceed max_errors={args.max_errors}") from exc

    summary = BatchSummary(total=len(samples), ok=len(metrics_rows), failed=failed)
    return metrics_rows, summary


def finalize_metrics(metrics_rows: list[dict[str, Any]], *, layout: OutputLayout) -> tuple[Path, Path]:
    thresholds = fit_outlier_thresholds(metrics_rows, config=OutlierRuleConfig())
    annotated_rows, review_rows = mark_anomalies(metrics_rows, thresholds, config=OutlierRuleConfig())

    metrics_path = export_metrics_csv(annotated_rows, layout.metrics_csv_path)
    review_path = export_review_csv(review_rows, layout.review_csv_path)
    return metrics_path, review_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config, overrides=to_overrides(args))

    output_dir = Path(config.engineering.io.output_dir)
    layout = build_output_layout(
        output_dir,
        snapshot_filename=config.engineering.output.snapshot_filename,
        metrics_filename=config.engineering.output.unsup_metrics_filename,
    )
    ensure_output_layout(layout, save_intermediate=config.engineering.output.save_intermediate)
    logger = setup_logger(layout, level=config.engineering.logging.level)

    export_config_snapshot(
        config,
        layout.snapshot_path,
        runtime_context={
            "cli": "scripts/run_infer.py",
            "config_path": args.config,
            "pipeline": args.pipeline,
            "dry_run": args.dry_run,
        },
    )

    samples = discover_batch_samples(
        config.engineering.io.input_dir,
        input_mode=args.input_mode,
        recursive=bool(args.recursive or config.engineering.io.recursive),
        input_glob=str(config.engineering.io.input_glob),
    )

    logger.info("discovered_samples=%d", len(samples))
    if args.dry_run:
        logger.info("dry_run enabled, stop before inference loop")
        return

    metrics_rows, summary = run_batch_loop(samples, config=config, args=args, layout=layout, logger=logger)
    logger.info("summary total=%d ok=%d failed=%d", summary.total, summary.ok, summary.failed)

    metrics_path, review_path = finalize_metrics(metrics_rows, layout=layout)
    logger.info("saved metrics=%s review=%s", metrics_path, review_path)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    if a.dtype == np.uint8:
        return a
    min_v = float(np.min(a))
    max_v = float(np.max(a))
    if max_v > min_v:
        a = (a - min_v) / (max_v - min_v)
    else:
        a = np.zeros_like(a, dtype=np.float32)
    return np.clip(a * 255.0, 0.0, 255.0).astype(np.uint8)


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


def _exts_from_glob(input_glob: str) -> tuple[str, ...]:
    # 支持 "*.png" 形式；复杂 glob 回退到默认后缀集合。
    g = input_glob.strip()
    if g.startswith("*.") and len(g) > 2 and "*" not in g[1:] and "?" not in g:
        return (g[1:].lower(),)
    return DiscoverOptions().exts


if __name__ == "__main__":
    main()
