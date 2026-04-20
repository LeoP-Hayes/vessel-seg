# 双尺度融合流水线设计（接口版）

## 目标
在单尺度 pipeline 基础上扩展两尺度推理：
- s0: 原图
- s1: 下采样图

两尺度独立推理后，s1 结果上采样对齐到原图尺寸，再做 OR 融合。

## 主函数
- `run_dual_scale_pipeline(image, params)`
- 输入：`image` 为 `(H, W)` 灰度 `np.ndarray`
- 输出：`(fused_mask, intermediate_dict)`

## 尺寸与对齐策略
- 下采样：`downsample_image(..., interpolation="area")`
- 上采样二值图：`upsample_binary_mask_to_shape(..., interpolation="nearest")`
- 上采样后二值化：默认开启 `binarize_after_upsample=True`
- 对齐锚点：最终尺寸严格对齐原图 `(H, W)`

## 每尺度参数读取
- `line_length_per_scale[i] -> tophat.line_length`
- `mean_filter_size_per_scale[i] -> illumination.window_size`
- 通过 `build_single_scale_params_for_level` 生成每尺度独立参数

## 最终融合
- `fuse_masks_or(mask_s0, mask_s1_up)`
- 输出 dtype 支持 `uint8|bool`，默认 `uint8`

## 工程默认策略（论文未明确）
- `scales` 固定两项且第一项必须 `1.0`
- 第二尺度在 `(0,1)` 区间
- `mean_filter_size_per_scale` 必须奇数且 `>=3`
- 二值图上采样插值固定 `nearest`
- 上采样后可选再二值化（默认启用）
