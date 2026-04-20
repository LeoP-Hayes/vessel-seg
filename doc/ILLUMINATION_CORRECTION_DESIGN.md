# illumination_correction 模块设计（接口版）

## 1. 公式与接口
目标公式：`Ieq = Is / (mean_filter(Is, N) + c)`

主函数：`illumination_correction(image, window_size, c, ...)`

当前阶段仅提供：
- 函数签名
- 参数/输入校验
- 异常约定
- 实现注意点

不包含真实滤波与除法计算。

## 2. 输入输出约束
输入：
- `image`: 2D 灰度 `np.ndarray`，推荐 `(512, 512)`

输出（实现阶段需满足）：
- 尺寸与输入一致
- 默认输出 `float32`
- 输出范围策略由 `out_range` 决定：`none | clip01 | minmax01`

## 3. 参数约束
- `window_size`：奇数且 `>=3`
- `c`：`>0`
- `eps`：`>0`

## 4. 数值稳定与边界策略（实现阶段）
- 分母项：`mean + c` 后再 `max(eps)`
- 防止 `NaN/Inf`：统一清洗
- 边界模式默认 `reflect`，备选 `replicate/constant`

## 5. 可插入主流程性
该模块可直接接在每尺度图像预处理后，输出传递给 top-hat 阶段。

模块文件：`src/vessel_reproduction/illumination_correction.py`
测试文件：`tests/test_illumination_correction_interfaces.py`
