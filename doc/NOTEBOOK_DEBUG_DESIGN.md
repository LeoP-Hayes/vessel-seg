# Notebook 调参与可视化设计（接口版）

## 章节结构
- `00_setup`
- `01_load_one_sample`
- `02_run_single_scale_debug`
- `03_run_dual_scale_debug`
- `04_visualization_panels`
- `05_parameter_sweep_compare`
- `06_export_and_review`

## 中间结果面板
默认展示：
- `input`
- `illumination_corrected`
- `tophat_fused`
- `threshold_binary`
- `denoised_binary`
- `overlay`

可选：方向响应面板（`tophat_directional`）

## 核心接口
- `make_overlay(image, mask, ...) -> np.ndarray`
- `run_once(run_cfg, pipeline_params) -> dict`
- `compare_param_sets(image, param_sets) -> list[dict]`
- `show_debug_panels(result, show_directional=False)`
- `show_comparison_grid(results, columns=3)`
- `show_metrics_table(results)`
- `export_notebook_artifacts(result, output_dir, prefix)`

## 一键重跑策略
通过 `run_once` 统一入口：
1. 读取单图
2. 调 single/dual pipeline
3. 生成 overlay
4. 返回可视化结果包

参数修改后仅重跑一个 cell，即可刷新全部面板。

## 展示逻辑
- 先单图完整链路（便于人工抽检）
- 再多参数并排对比（便于快速调参）
- 最后导出关键图与参数快照（便于复现）
