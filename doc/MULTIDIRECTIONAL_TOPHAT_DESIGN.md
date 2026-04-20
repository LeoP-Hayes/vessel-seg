# 多方向改进 Top-hat 模块设计（接口版）

## 目标
提供可独立测试、可复用于单尺度/双尺度的多方向改进 top-hat 接口。
当前阶段仅实现：
- 函数签名
- 参数与输入校验
- 方向与结构元素离散化生成
- 融合接口
- 中间结果接口占位

不实现完整形态学主流程。

## 接口总览
- `validate_tophat_params`
- `generate_direction_specs`
- `generate_line_structuring_element`
- `compute_modified_tophat_response`
- `fuse_directional_responses`
- `run_multidirectional_tophat`
- `serialize_tophat_intermediate`
- `intermediate_to_rows`

## 关键约束
- `num_directions >= 2`
- `line_length >= 2`
- 偶数 `line_length` 自动提升为 `line_length + 1`
- 输入图像必须是 2D 灰度 `np.ndarray`
- `save_directional=True` 依赖 `save_intermediate=True`

## 假设与未明确细节
- 改进 top-hat 固定公式：`Ieq - min(open(close(Ieq, S), S), Ieq)`。
- `M=9` 默认方向集合包含 `180°`：`[0,22.5,...,180]`。
- 线结构元素离散化：中心锚点 + 端点四舍五入 + Bresenham 连线。
- 厚度目前仅支持 `1` 像素。
- 边界模式默认 `reflect`。

## 中间结果
`TopHatIntermediate` 支持：
- `angles_deg`
- `fused_response`
- `selems`（可选）
- `directional_responses`（可选）
- `directional_backgrounds`（可选）
