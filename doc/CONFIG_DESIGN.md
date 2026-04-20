# 配置系统与快照机制设计

## 1) default.yaml 覆盖范围
`config/default.yaml` 已覆盖 PLAN.md 的关键参数：
- `scales`
- `M` (`num_directions`)
- `L_per_scale` (`line_length_per_scale`)
- `N_per_scale` (`mean_filter_size_per_scale`)
- `c` (`epsilon_c`)
- `adaptive_block_size`
- `adaptive_C`
- `area_min`

并补充了工程参数：I/O、日志、异常策略、输出开关。

## 2) Python 配置接口
- `load_config(config_path, overrides)`：加载 YAML + dot-key 覆盖，并转换为 `AppConfig`。
- `export_config_snapshot(config, output_path, runtime_context)`：导出快照。

数据结构定义在：`src/vessel_reproduction/config_schema.py`。

## 3) 运行时快照规则
快照内容固定包含：
- `meta`
- `algorithm`
- `engineering`
- `runtime_context`（如 CLI 入口、配置路径、时间戳）
- `param_taxonomy`（每个参数属于算法/工程）
- `config_digest`（SHA-256）

## 4) 参数分类
- 算法参数：`algorithm.*`
- 工程参数：`engineering.*`
- 细粒度映射：`PARAM_TAXONOMY`

## 5) CLI 复用方式
`scripts/run_infer.py` 已提供最小骨架：
1. 读取 `--config`
2. 合并 CLI 覆盖项
3. 在输出目录写入快照

后续可直接在 `main()` 中接入算法流水线。
