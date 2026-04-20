from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence

import cv2
import numpy as np


ImageBackend = Literal["opencv", "pillow"]
OutputDType = Literal["uint8", "float32"]
InputMode = Literal["auto", "single_dir", "split_dirs"]
SortMode = Literal["name", "path", "mtime"]
SplitName = Literal["train", "val", "test", "all"]
ReadStatus = Literal["ok", "failed", "skipped"]
OnErrorPolicy = Literal["raise", "skip", "record"]
ErrorCode = Literal[
    "PATH_NOT_FOUND",
    "UNSUPPORTED_EXT",
    "PERMISSION_DENIED",
    "DECODE_FAILED",
    "INVALID_SHAPE",
    "EMPTY_IMAGE",
    "INTERNAL_ERROR",
]


class LoggerLike(Protocol):
    """最小日志协议，便于后续接入 logging.Logger 或自定义日志器。"""

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        ...

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        ...

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        ...


@dataclass(frozen=True)
class ReadImageOptions:
    backend: ImageBackend = "opencv"
    out_dtype: OutputDType = "uint8"
    normalize: bool = False
    strict: bool = True


@dataclass(frozen=True)
class ReadImageMeta:
    path: str
    width: int
    height: int
    channels_original: int
    dtype_original: str
    dtype_output: str
    min_value: float
    max_value: float


@dataclass(frozen=True)
class DiscoverOptions:
    input_mode: InputMode = "auto"
    exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
    recursive: bool = False
    followlinks: bool = False
    sort: SortMode = "path"


@dataclass(frozen=True)
class SampleCandidate:
    sample_id: str
    split: SplitName
    input_root: str
    rel_path: str
    abs_path: str
    filename: str
    suffix: str
    file_size_bytes: int
    mtime_utc: str


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    split: SplitName
    input_root: str
    rel_path: str
    abs_path: str
    filename: str
    suffix: str
    file_size_bytes: int
    mtime_utc: str
    read_status: ReadStatus
    error_code: ErrorCode | None
    error_message: str | None
    width: int | None
    height: int | None
    dtype_output: str | None
    min_value: float | None
    max_value: float | None
    checksum_sha1: str | None
    created_at_utc: str


@dataclass(frozen=True)
class ManifestBuildOptions:
    read_image: bool = True
    include_checksum: bool = False


@dataclass(frozen=True)
class ProcessBatchOptions:
    on_error: OnErrorPolicy = "record"
    max_errors: int = 100


@dataclass(frozen=True)
class BatchSummary:
    total: int
    ok: int
    failed: int
    skipped: int


class DataPipelineError(RuntimeError):
    """批处理级错误：例如错误数超过阈值、输入目录非法等。"""


class SampleReadError(RuntimeError):
    """单样本读取错误，携带 error_code 便于写入 manifest 与日志。"""

    def __init__(self, message: str, *, error_code: ErrorCode) -> None:
        super().__init__(message)
        self.error_code = error_code


DEFAULT_MANIFEST_FIELDS: tuple[str, ...] = (
    "sample_id",
    "split",
    "input_root",
    "rel_path",
    "abs_path",
    "filename",
    "suffix",
    "file_size_bytes",
    "mtime_utc",
    "read_status",
    "error_code",
    "error_message",
    "width",
    "height",
    "dtype_output",
    "min_value",
    "max_value",
    "checksum_sha1",
    "created_at_utc",
)


def read_grayscale_image(
    path: str | Path,
    *,
    options: ReadImageOptions | None = None,
) -> tuple[np.ndarray, ReadImageMeta]:
    """读取单图并统一输出灰度 ndarray(H,W)。"""
    opts = options or ReadImageOptions()
    p = Path(path)

    if not p.exists():
        raise SampleReadError(f"image path not found: {p}", error_code="PATH_NOT_FOUND")
    if not p.is_file():
        raise SampleReadError(f"not a file: {p}", error_code="PATH_NOT_FOUND")

    if opts.backend != "opencv":
        raise SampleReadError("only opencv backend is currently implemented", error_code="INTERNAL_ERROR")

    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise SampleReadError(f"failed to decode image: {p}", error_code="DECODE_FAILED")

    channels_original = 1
    if raw.ndim == 2:
        gray = raw
        channels_original = 1
    elif raw.ndim == 3 and raw.shape[2] == 3:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        channels_original = 3
    elif raw.ndim == 3 and raw.shape[2] == 4:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        channels_original = 4
    else:
        raise SampleReadError(f"invalid image shape: {raw.shape}", error_code="INVALID_SHAPE")

    if gray.size == 0:
        raise SampleReadError("empty image", error_code="EMPTY_IMAGE")

    out: np.ndarray
    if opts.out_dtype == "float32":
        out = gray.astype(np.float32)
        if opts.normalize:
            max_val = float(np.max(out))
            if max_val > 1.0:
                out = np.clip(out / 255.0, 0.0, 1.0)
    elif opts.out_dtype == "uint8":
        if gray.dtype == np.uint8:
            out = gray
        else:
            out = np.clip(gray, 0, 255).astype(np.uint8)
    else:
        raise SampleReadError(f"unsupported out_dtype: {opts.out_dtype}", error_code="INTERNAL_ERROR")

    h, w = out.shape
    meta = ReadImageMeta(
        path=str(p),
        width=int(w),
        height=int(h),
        channels_original=channels_original,
        dtype_original=str(gray.dtype),
        dtype_output=str(out.dtype),
        min_value=float(np.min(out)),
        max_value=float(np.max(out)),
    )
    return out, meta


def discover_samples(
    input_root: str | Path,
    *,
    options: DiscoverOptions | None = None,
) -> list[SampleCandidate]:
    """遍历目录并返回候选样本。"""
    opts = options or DiscoverOptions()
    root = Path(input_root)
    if not root.exists() or not root.is_dir():
        raise DataPipelineError(f"input_root is not a valid directory: {root}")

    split_dirs = {name: root / name for name in ("train", "val", "test")}
    has_split_dir = any(p.exists() and p.is_dir() for p in split_dirs.values())

    if opts.input_mode == "split_dirs":
        mode: InputMode = "split_dirs"
    elif opts.input_mode == "single_dir":
        mode = "single_dir"
    else:
        mode = "split_dirs" if has_split_dir else "single_dir"

    exts = {e.lower() for e in opts.exts}
    candidates: list[SampleCandidate] = []

    if mode == "split_dirs":
        for split in ("train", "val", "test"):
            split_root = split_dirs[split]
            if not split_root.exists() or not split_root.is_dir():
                continue
            candidates.extend(_collect_from_dir(root, split_root, split=split, exts=exts, recursive=opts.recursive))
    else:
        candidates.extend(_collect_from_dir(root, root, split="all", exts=exts, recursive=opts.recursive))

    if opts.sort == "name":
        candidates.sort(key=lambda x: x.filename)
    elif opts.sort == "mtime":
        candidates.sort(key=lambda x: x.mtime_utc)
    else:
        candidates.sort(key=lambda x: x.rel_path)

    return candidates


def build_manifest(
    discovered: Sequence[SampleCandidate],
    *,
    options: ManifestBuildOptions | None = None,
    logger: LoggerLike | None = None,
) -> list[ManifestRecord]:
    """基于候选样本构建 manifest 记录（可选执行读图补充尺寸与强度统计）。"""
    opts = options or ManifestBuildOptions()
    records: list[ManifestRecord] = []

    for item in discovered:
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            width: int | None = None
            height: int | None = None
            dtype_output: str | None = None
            min_value: float | None = None
            max_value: float | None = None
            checksum_sha1: str | None = None

            if opts.read_image:
                img, meta = read_grayscale_image(item.abs_path)
                width = meta.width
                height = meta.height
                dtype_output = meta.dtype_output
                min_value = meta.min_value
                max_value = meta.max_value
                if opts.include_checksum:
                    checksum_sha1 = hashlib.sha1(img.tobytes()).hexdigest()

            record = ManifestRecord(
                sample_id=item.sample_id,
                split=item.split,
                input_root=item.input_root,
                rel_path=item.rel_path,
                abs_path=item.abs_path,
                filename=item.filename,
                suffix=item.suffix,
                file_size_bytes=item.file_size_bytes,
                mtime_utc=item.mtime_utc,
                read_status="ok",
                error_code=None,
                error_message=None,
                width=width,
                height=height,
                dtype_output=dtype_output,
                min_value=min_value,
                max_value=max_value,
                checksum_sha1=checksum_sha1,
                created_at_utc=created_at,
            )
            records.append(record)
        except SampleReadError as exc:
            if logger:
                logger.warning("build_manifest failed sample_id=%s err=%s", item.sample_id, exc)
            records.append(
                ManifestRecord(
                    sample_id=item.sample_id,
                    split=item.split,
                    input_root=item.input_root,
                    rel_path=item.rel_path,
                    abs_path=item.abs_path,
                    filename=item.filename,
                    suffix=item.suffix,
                    file_size_bytes=item.file_size_bytes,
                    mtime_utc=item.mtime_utc,
                    read_status="failed",
                    error_code=exc.error_code,
                    error_message=str(exc),
                    width=None,
                    height=None,
                    dtype_output=None,
                    min_value=None,
                    max_value=None,
                    checksum_sha1=None,
                    created_at_utc=created_at,
                )
            )

    return records


def process_batch(
    samples: Sequence[SampleCandidate],
    *,
    options: ProcessBatchOptions | None = None,
    logger: LoggerLike | None = None,
) -> tuple[BatchSummary, list[ManifestRecord]]:
    """批处理入口：应用异常策略并返回汇总与 manifest。"""
    opts = options or ProcessBatchOptions()
    failed_count = 0
    skipped_count = 0
    records: list[ManifestRecord] = []

    for item in samples:
        try:
            rec = build_manifest([item], options=ManifestBuildOptions(read_image=True, include_checksum=False), logger=logger)[0]
            if rec.read_status == "failed":
                raise SampleReadError(rec.error_message or "read failed", error_code=rec.error_code or "INTERNAL_ERROR")
            records.append(rec)
        except SampleReadError as exc:
            failed_count += 1
            if opts.on_error == "raise":
                raise DataPipelineError(f"batch failed at sample_id={item.sample_id}: {exc}") from exc
            if opts.on_error == "skip":
                skipped_count += 1
                continue
            records.append(
                ManifestRecord(
                    sample_id=item.sample_id,
                    split=item.split,
                    input_root=item.input_root,
                    rel_path=item.rel_path,
                    abs_path=item.abs_path,
                    filename=item.filename,
                    suffix=item.suffix,
                    file_size_bytes=item.file_size_bytes,
                    mtime_utc=item.mtime_utc,
                    read_status="failed",
                    error_code=exc.error_code,
                    error_message=str(exc),
                    width=None,
                    height=None,
                    dtype_output=None,
                    min_value=None,
                    max_value=None,
                    checksum_sha1=None,
                    created_at_utc=datetime.now(timezone.utc).isoformat(),
                )
            )
            if failed_count > opts.max_errors:
                raise DataPipelineError(f"failed samples exceed max_errors={opts.max_errors}")

    ok_count = sum(1 for r in records if r.read_status == "ok")
    summary = BatchSummary(total=len(samples), ok=ok_count, failed=failed_count, skipped=skipped_count)
    return summary, records


def manifest_to_rows(records: Sequence[ManifestRecord]) -> list[Mapping[str, Any]]:
    """将 dataclass 记录转换为字典行，供 CSV 写出层复用。"""
    return [asdict(r) for r in records]


def _collect_from_dir(
    input_root: Path,
    search_root: Path,
    *,
    split: SplitName,
    exts: set[str],
    recursive: bool,
) -> list[SampleCandidate]:
    pattern = "**/*" if recursive else "*"
    out: list[SampleCandidate] = []

    for p in search_root.glob(pattern):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        st = p.stat()
        rel = p.relative_to(input_root).as_posix()
        sample_id = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
        out.append(
            SampleCandidate(
                sample_id=sample_id,
                split=split,
                input_root=str(input_root),
                rel_path=rel,
                abs_path=str(p.resolve()),
                filename=p.name,
                suffix=p.suffix.lower(),
                file_size_bytes=int(st.st_size),
                mtime_utc=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
            )
        )
    return out
