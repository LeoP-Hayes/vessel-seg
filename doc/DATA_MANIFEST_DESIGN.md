# 数据读取与批处理样本清单模块设计（接口版）

## 目标
仅定义接口、输入输出与异常策略，不实现具体业务逻辑。

## 1. 图像读取接口
函数：`read_grayscale_image(path, options)`

输入：
- `path`: `str | Path`
- `options.backend`: `opencv | pillow`
- `options.out_dtype`: `uint8 | float32`
- `options.normalize`: 是否归一化到 `[0,1]`
- `options.strict`: 严格模式

输出：
- `gray`: `np.ndarray`，约束 `shape=(H,W)`
- `meta`: `ReadImageMeta`

异常：
- 统一抛 `SampleReadError(error_code=...)`

## 2. 目录遍历接口
函数：`discover_samples(input_root, options)`

输入模式：
- `split_dirs`: `train/val/test`
- `single_dir`: 全部标记 `split=all`
- `auto`: 自动判定

输出：
- `list[SampleCandidate]`

字段包含：
- 路径信息（绝对/相对）
- 文件信息（后缀、大小、mtime）
- `sample_id`
- `split`

## 3. Manifest 字段设计（CSV 友好）
`ManifestRecord` 字段：
- `sample_id`
- `split`
- `input_root`
- `rel_path`
- `abs_path`
- `filename`
- `suffix`
- `file_size_bytes`
- `mtime_utc`
- `read_status`
- `error_code`
- `error_message`
- `width`
- `height`
- `dtype_output`
- `min_value`
- `max_value`
- `checksum_sha1`
- `created_at_utc`

## 4. 异常与日志策略
错误码：
- `PATH_NOT_FOUND`
- `UNSUPPORTED_EXT`
- `PERMISSION_DENIED`
- `DECODE_FAILED`
- `INVALID_SHAPE`
- `EMPTY_IMAGE`
- `INTERNAL_ERROR`

批处理策略（`process_batch`）：
- `on_error=raise`: 首错即停
- `on_error=skip`: 跳过异常样本，不写失败记录
- `on_error=record`: 写失败记录并继续，超过 `max_errors` 抛 `DataPipelineError`

日志建议：
- `INFO`: 启动参数、处理汇总
- `WARNING`: 可恢复异常样本
- `ERROR`: 不可恢复异常、超过阈值终止

## 5. 模块导出
模块文件：`src/vessel_reproduction/data_manifest.py`

核心接口：
- `read_grayscale_image`
- `discover_samples`
- `build_manifest`
- `process_batch`
- `manifest_to_rows`
