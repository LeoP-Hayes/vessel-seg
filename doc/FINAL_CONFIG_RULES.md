# final.yaml 固化规则（提纲版）

## 1. 目标
- 在一次完整批处理结束后，生成可交接、可复现的最终参数与产物快照。

## 2. 落盘位置与生命周期
- 主文件：`outputs/<run_id>/final.yaml`
- 最新引用：`outputs/latest_final.yaml`（可软链或复制）
- 规则：历史 `final.yaml` 不覆盖，只新增 run 版本

## 3. 合并优先级
1. `config/default.yaml`
2. CLI 覆盖项
3. 运行时修正后的 effective 参数

## 4. 必填区块（待实现时必须覆盖）
- `meta`
- `algorithm`
- `engineering`
- `effective_params`
- `artifacts`
- `quality_gate`
- `digests`
- `assumptions_ref`

## 5. 字段模板（待填）
```yaml
meta:
  run_id: TODO
  created_at_utc: TODO
  project_version: TODO
algorithm: {}
engineering: {}
effective_params:
  line_length_per_scale: TODO
  mean_filter_size_per_scale: TODO
  adaptive_block_size: TODO
  adaptive_C: TODO
  area_min: TODO
artifacts:
  masks_dir: TODO
  overlays_dir: TODO
  intermediate_dir: TODO
  metrics_unsup_csv: TODO
  review_list_csv: TODO
  run_log: TODO
quality_gate:
  total_samples: TODO
  ok_samples: TODO
  failed_samples: TODO
  anomaly_samples: TODO
  outlier_rule: TODO
digests:
  config_digest: TODO
  assumptions_digest: TODO
  metrics_digest: TODO
assumptions_ref:
  file: ASSUMPTIONS.md
  active_ids: TODO
```

## 6. 校验清单
- [ ] 与 `run_config_snapshot.yaml` 一致性校验
- [ ] 所有 artifact 路径可访问
- [ ] effective 参数已写入“最终生效值”
- [ ] 异常规则配置已固化到 `quality_gate`
- [ ] digest 字段完整

## 7. 与交接文档联动
- `ASSUMPTIONS.md`：提供假设追踪来源
- `README.md`：提供操作入口与产物解释
