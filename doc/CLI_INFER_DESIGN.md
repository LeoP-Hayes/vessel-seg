# CLI 端到端推理入口设计（接口版）

## 命令行参数
`run_infer.py` 核心参数：
- `--config`
- `--input_dir`, `--output_dir`
- `--pipeline {single,dual}`
- `--input_mode {auto,single_dir,split_dirs}`
- `--recursive`
- `--save_intermediate`, `--save_directional`
- `--skip_on_error`, `--max_errors`
- `--log_level`
- `--dry_run`

## 程序入口与调用顺序
1. `parse_args`
2. `load_config` + `to_overrides`
3. 构建并创建输出目录布局
4. 初始化日志器
5. 落盘 `run_config_snapshot.yaml`
6. 发现输入样本（目录级批处理）
7. 批处理主循环：逐样本读取/推理/导出/统计
8. 失败样本记录（jsonl）
9. 批次结束后执行异常阈值拟合与标记
10. 落盘 `metrics_unsup.csv` 与 `review_list.csv`

## 输出目录结构与命名
- `masks/`：`{rel_stem}_mask.png`
- `overlays/`：`{rel_stem}_overlay.png`
- `intermediate/`：按阶段中间图（可选）
- `logs/run.log`
- `logs/failed_samples.jsonl`
- `run_config_snapshot.yaml`
- `metrics_unsup.csv`
- `review_list.csv`

其中 `rel_stem` 为相对路径去后缀后，将 `/` 替换为 `__`。

## 异常与日志机制
- 单样本异常写入 `failed_samples.jsonl`
- `skip_on_error=False`：首错终止
- `skip_on_error=True`：跳过异常并继续
- 当失败数超过 `max_errors`：强制终止

## 与 PLAN.md 一致性
- 支持目录级批处理
- 产物覆盖：`mask/overlay/intermediate/metrics_unsup.csv/run_config_snapshot.yaml`
- 统计与复核清单链路闭环（`review_list.csv`）
