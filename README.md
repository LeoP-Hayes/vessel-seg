# vessel_reproduction

基于无监督形态学流程的 FFA（荧光素眼底血管造影）血管提取项目。

本仓库已经实现从目录级数据读取到单/双/三尺度推理、结果导出、无监督统计与异常样本标记的完整 CLI 流程。

## 目录

- [1. 项目目标](#1-项目目标)
- [2. 核心能力](#2-核心能力)
- [3. 项目结构与模块职责](#3-项目结构与模块职责)
- [4. 环境安装](#4-环境安装)
- [5. 数据组织方式](#5-数据组织方式)
- [6. 快速开始](#6-快速开始)
- [7. CLI 详解（合法 options 全量）](#7-cli-详解合法-options-全量)
- [8. 配置系统与参数分类](#8-配置系统与参数分类)
- [9. 推理输出说明](#9-推理输出说明)
- [10. 算法流程说明](#10-算法流程说明)
- [11. 测试与验证](#11-测试与验证)
- [12. 已知限制与后续扩展](#12-已知限制与后续扩展)

## 1. 项目目标

给定 FFA 灰度图像，输出血管二值分割结果，并提供：

- 可复现的运行配置快照
- 批量无监督质量统计 (`metrics_unsup.csv`)
- 自动异常样本清单 (`review_list.csv`)

## 2. 核心能力

- 单图读取统一为灰度 `ndarray(H, W)`
- 单尺度分割流水线（光照校正 -> 多方向改进 top-hat -> 自适应阈值 -> 面积去噪）
- 双尺度推理与 OR 融合
- 三尺度推理与 OR 融合
- 批处理容错（跳过失败样本、错误上限）
- 配置快照导出（含参数分类与 digest）

## 3. 项目结构与模块职责

```text
vessel_reproduction/
├── config/
│   ├── default.yaml               # 默认配置
│   └── final.yaml.template        # 最终参数固化模板
├── data/                          # 仅保留目录结构占位（不含私有数据）
├── scripts/
│   └── run_infer.py               # 端到端 CLI 入口
├── src/vessel_reproduction/
│   ├── config_schema.py           # 配置 dataclass + 参数分类
│   ├── config_loader.py           # 配置加载/覆盖/快照导出
│   ├── data_manifest.py           # 读图、样本发现、manifest 记录
│   ├── illumination_correction.py # 光照校正 Ieq = Is/(mean+c)
│   ├── multidirectional_tophat.py # 多方向改进 top-hat
│   ├── single_scale_pipeline.py   # 单尺度流程
│   ├── dual_scale_pipeline.py     # 双尺度流程
│   ├── unsupervised_metrics.py    # 无监督统计 + 异常标记
│   └── notebook_debug.py          # Notebook 可视化接口（部分占位）
└── tests/                         # 单元测试
```

### 模块职责简表

- `config_schema.py`：定义算法参数与工程参数结构，提供 `PARAM_TAXONOMY`。
- `config_loader.py`：读取 YAML、应用 overrides、导出 `run_config_snapshot.yaml`。
- `data_manifest.py`：扫描 `single_dir`/`train-val-test`，读取灰度图，构建 manifest。
- `illumination_correction.py`：执行均值滤波背景校正与归一化。
- `multidirectional_tophat.py`：生成方向结构元素，计算方向响应并逐像素 `max` 融合。
- `single_scale_pipeline.py`：单尺度全流程与中间结果组织。
- `dual_scale_pipeline.py`：多尺度（含双/三尺度）独立推理、上采样对齐、OR 融合。
- `unsupervised_metrics.py`：导出面积比、连通域、骨架等指标，并做 IQR/分位数异常标记。
- `scripts/run_infer.py`：批处理主循环、日志、产物落盘、异常容错。

## 4. 环境安装

建议 Python 3.10+。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy opencv-python pyyaml
```

如果用于 Notebook 展示，可按需安装：

```bash
pip install matplotlib
```

## 5. 数据组织方式

支持两类输入：

1. 单目录模式（`single_dir`）：

```text
<your_data_dir>/
  0.png
  1.png
  ...
```

2. 划分目录模式（`split_dirs`）：

```text
<your_data_dir>/
  train/*.png
  val/*.png
  test/*.png
```

`--input_mode auto` 会自动判断：若存在 `train/val/test` 目录则按 `split_dirs`，否则按 `single_dir`。

> 注意：仓库中的 `data/` 目录仅保留结构占位（`train/val/test`），不包含真实图像文件。

## 6. 快速开始

### 6.1 双尺度批处理

```bash
python3 scripts/run_infer.py \
  --config config/default.yaml \
  --input_dir <your_data_dir> \
  --output_dir outputs/run_dual \
  --pipeline dual \
  --input_mode single_dir \
  --recursive \
  --skip_on_error \
  --max_errors 5
```

### 6.2 单尺度批处理

```bash
python3 scripts/run_infer.py \
  --config config/default.yaml \
  --input_dir <your_data_dir> \
  --output_dir outputs/run_single \
  --pipeline single
```

### 6.3 三尺度批处理

```bash
python3 scripts/run_infer.py \
  --config config/default.yaml \
  --input_dir <your_data_dir> \
  --output_dir outputs/run_triple \
  --pipeline triple
```

### 6.4 仅检查配置与样本发现（不执行推理）

```bash
python3 scripts/run_infer.py --config config/default.yaml --dry_run
```

## 7. CLI 详解（合法 options 全量）

入口脚本：`scripts/run_infer.py`

| Option | 类型 | 默认值 | 合法取值 | 说明 |
|---|---|---:|---|---|
| `--config` | `str` | `config/default.yaml` | 任意可读 YAML 路径 | 主配置文件 |
| `--input_dir` | `str` | `None` | 任意目录路径 | 覆盖 `engineering.io.input_dir` |
| `--output_dir` | `str` | `None` | 任意输出目录路径 | 覆盖 `engineering.io.output_dir` |
| `--pipeline` | `str` | `dual` | `single` / `dual` / `triple` | 推理模式 |
| `--input_mode` | `str` | `auto` | `auto` / `single_dir` / `split_dirs` | 样本发现模式 |
| `--recursive` | `flag` | `False` | 出现即启用 | 递归遍历目录；与配置 `io.recursive` 取 OR |
| `--save_intermediate` | `flag` | `False` | 出现即启用 | 覆盖 `engineering.output.save_intermediate=true` |
| `--save_directional` | `flag` | `False` | 出现即启用 | 在 pipeline debug 中请求方向级中间结果 |
| `--skip_on_error` | `flag` | `False` | 出现即启用 | 覆盖 `engineering.robustness.skip_on_error=true` |
| `--max_errors` | `int` | `50` | `>=0` 整数 | 覆盖最大可容忍失败样本数 |
| `--log_level` | `str` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` | 覆盖日志等级 |
| `--dry_run` | `flag` | `False` | 出现即启用 | 仅做配置加载和样本发现，不跑推理 |

### 7.1 常见组合示例

1. 强制按 `train/val/test` 读取：

```bash
python3 scripts/run_infer.py --input_mode split_dirs
```

2. 单目录 + 保存中间结果：

```bash
python3 scripts/run_infer.py --input_mode single_dir --save_intermediate
```

3. 失败即停（不跳过）：

```bash
python3 scripts/run_infer.py --max_errors 0
```

## 8. 配置系统与参数分类

默认配置文件：`config/default.yaml`

### 8.1 参数优先级

- 基础：`default.yaml`
- 覆盖：CLI 部分参数（如 `--input_dir`）
- 结果：运行快照 `run_config_snapshot.yaml`

### 8.2 算法参数（`algorithm.*`）

- `algorithm.pyramid.scales`
- `algorithm.morphology.num_directions`
- `algorithm.morphology.line_length_per_scale`
- `algorithm.illumination.mean_filter_size_per_scale`
- `algorithm.illumination.epsilon_c`
- `algorithm.threshold.method`
- `algorithm.threshold.adaptive_block_size`
- `algorithm.threshold.adaptive_C`
- `algorithm.postprocess.area_min`
- `algorithm.fusion.method`
- `algorithm.preprocess.invert_intensity`

默认三尺度基线（可改）：
- `scales: [1.0, 0.5, 0.25]`
- `line_length_per_scale: [6, 3, 2]`
- `mean_filter_size_per_scale: [7, 5, 3]`

### 8.3 工程参数（`engineering.*`）

- 输入输出：`engineering.io.*`
- 产物保存：`engineering.output.*`
- 运行资源：`engineering.runtime.*`
- 稳定性：`engineering.robustness.*`
- 日志：`engineering.logging.*`

### 8.4 `input_glob` 当前行为说明

- 推荐写法：`"*.png"`、`"*.jpg"` 等。
- 当前实现会从简单 `*.ext` 形式提取后缀过滤。
- 复杂 glob（如多个通配组合）会回退到默认后缀集合：`.png/.jpg/.jpeg/.tif/.tiff/.bmp`。

## 9. 推理输出说明

以 `--output_dir outputs/run_dual` 为例：

```text
outputs/run_dual/
├── masks/                     # 二值分割图（0/255 PNG）
├── overlays/                  # 叠加可视化图
├── intermediate/              # 可选中间结果（启用 save_intermediate）
├── logs/
│   ├── run.log                # 运行日志
│   └── failed_samples.jsonl   # 失败样本记录
├── run_config_snapshot.yaml   # 配置快照（含 taxonomy + digest）
├── metrics_unsup.csv          # 无监督统计结果
└── review_list.csv            # 异常样本复核清单
```

## 10. 算法流程说明

### 10.1 单尺度

1. `illumination_correction`: `Ieq = Is / (mean_filter(Is, N) + c)`
2. `run_multidirectional_tophat`: 多方向响应后取逐像素 `max`
3. `adaptive_threshold_single`: 自适应阈值二值化
4. `remove_small_components`: 小连通域去噪

### 10.2 双尺度

1. 原尺度 `s0=1.0` 与下采样尺度 `s1` 独立执行单尺度流程
2. `s1` 结果最近邻上采样并对齐到原图尺寸
3. 两尺度二值结果执行逐像素 OR 融合

### 10.3 三尺度

1. 在 `s0=1.0`、`s1`、`s2` 三个分辨率独立执行单尺度流程
2. 将 `s1/s2` 结果最近邻上采样到原图尺寸
3. 三尺度掩码逐像素 OR 融合输出

### 10.4 无监督质检

- 指标：面积比、连通域数量、最大连通域占比、骨架长度、分叉点/端点等
- 规则：优先 IQR，样本不足或 IQR 退化时回退分位数阈值
- 输出：`metrics_unsup.csv` + `review_list.csv`

## 11. 测试与验证

运行全部测试：

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

建议在改动核心算法后至少执行：

1. 全量单元测试
2. 一次自备数据目录端到端推理
3. 人工检查 `overlays/` 与 `review_list.csv`

## 12. 已知限制与后续扩展

### 当前限制

- `notebook_debug.py` 中部分展示函数仍为接口占位（不影响 CLI 主流程）。
- 暂未集成 ROI 裁剪与视盘/背景先验。
- `input_glob` 仅对简单 `*.ext` 形式做严格后缀提取。

### 建议扩展

- 增加 ROI 预处理与可配置开关
- 增加并行批处理（`num_workers`）
- 增加更丰富的导出报告（HTML/Markdown）
- 增加固定数据集回归基线（指标阈值 + 可视化快照）

---

如果你准备继续迭代，建议下一步优先做两件事：

1. 补齐 `notebook_debug.py` 占位函数，形成完整调参闭环。
2. 固化 `config/final.yaml` 与一组基线回归样本，确保后续改动可量化评估。
