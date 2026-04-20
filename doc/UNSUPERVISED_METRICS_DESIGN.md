# 无监督统计与异常样本标记模块设计（接口版）

## 目标
为批量质检提供：
- 每图无监督统计指标
- 基于 IQR/分位数的异常样本标记
- `metrics_unsup.csv` 与 `review_list.csv` 输出格式

与 `PLAN.md` 的“双轨验收”一致：
- 统计门禁（批量规则）
- 人工复核清单（review list）

## 每图指标定义
- `vessel_area_ratio`
- `connected_components_count`
- `largest_component_ratio`
- `skeleton_length_px`
- `mean_branch_degree`
- `branch_points_count`
- `endpoints_count`

## 异常规则
- 主规则：IQR
  - 区间：`[Q1-k*IQR, Q3+k*IQR]`
- 回退规则：Quantile
  - 在样本不足或 `IQR≈0` 时使用 `quantile_low/quantile_high`
- 评分：`anomaly_score = triggered_metrics_count`
- 进入复核：`anomaly_score >= score_threshold`

## CSV 字段
`metrics_unsup.csv`：
- `sample_id, split, rel_path, mask_path, height, width`
- `vessel_area_ratio, connected_components_count, largest_component_ratio`
- `skeleton_length_px, mean_branch_degree, branch_points_count, endpoints_count`
- `metrics_version, created_at_utc`

`review_list.csv`：
- `sample_id, split, rel_path`
- `anomaly_score, severity, triggered_metrics, rule_type`
- `threshold_snapshot, key_metric_values, created_at_utc`

## 依赖与近似
- skeleton 首选：`skimage`（`skeletonize`）
- 近似方案：`opencv_approx`（当 `skimage` 不可用）
- branch degree 约定：基于骨架图 8 邻域统计分叉点邻接度
