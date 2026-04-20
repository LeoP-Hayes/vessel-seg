# 单尺度分割流水线设计（接口版）

## 目标
串联：
1. illumination_correction
2. directional_tophat
3. adaptive_threshold
4. area_denoise

仅提供流程结构与函数关系，不实现完整业务逻辑。

## 主函数
- `run_single_scale_pipeline(image, params)`
- 输入：`image` 为 `(H,W)` 灰度 `np.ndarray`
- 输出：`(final_mask, intermediate_dict)`

## 阶段输入输出约束
- 输入图：`(H,W)`, `uint8|float32`
- illumination 输出：`(H,W)`, `float32`, 默认语义 `[0,1]`
- tophat 输出：`(H,W)`, `float32`
- threshold 输出：`(H,W)` binary mask（`uint8|bool`）
- denoise 输出：`(H,W)` binary mask（`uint8|bool`）

## 中间结果字典键
- `input`
- `illumination_corrected`
- `tophat_fused`
- `tophat_directional`
- `threshold_binary`
- `denoised_binary`
- `meta`

## 工程必须明确的默认策略（论文未明确）
- `threshold.block_size`：奇数且 `>=3`
- `threshold.method`：仅 `adaptive_mean`
- `denoise.connectivity`：默认 `8`
- `denoise.area_min`：`>=0`
- `mask` 编码默认可配置 `uint8|bool`
- `debug.include_directional=True` 时才返回方向级中间图
