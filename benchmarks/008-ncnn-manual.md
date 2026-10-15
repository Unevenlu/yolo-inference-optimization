# 008 - NCNN 推理 + 全手写前后处理管线

**日期**: 2026-10-15

## 目的

脱离 ultralytics 推理封装，手写端侧部署的完整管线——**letterbox 前处理 → NCNN 推理 → 输出解码 → NMS**，这是任何 C++/移动端部署都绕不开的代码。

> 说明：本机暂无 C++ 工具链（无 MSVC/MinGW/CMake），故用 NCNN 官方 Python 绑定（pyncnn）实现。管线算法逻辑与 C++ 版本 1:1 对应，后续移植只改语言不改逻辑。

## 管线四步（scripts/ncnn_manual_bench.py）

1. **letterbox**：等比缩放 + 114 灰边填充到 640×640，记录缩放比与 padding 供逆映射
2. **NCNN 推理**：`from_pixels` 构建输入 Mat → `/255` 归一化 → extractor 推理
3. **解码**：输出 (84, 8400) 转置 → 拆 4 坐标 + 80 类分数 → 每框取最大类 → xywh 转 xyxy → 反 letterbox 映射回原图
4. **手写 NMS**：按分数降序，逐框抑制 IoU > 0.45 的同类别框

## 正确性验证

手写管线与 ultralytics 封装在 bus.jpg 上**逐框完全一致**（4 人 + 1 巴士，置信度与坐标相同）——管线实现正确。

## 性能结果

| 引擎 | 推理耗时 | 说明 |
|------|----------|------|
| NCNN FP32（手写管线纯推理） | 48.2ms | pyncnn, yolov8n |
| NCNN FP32（ultralytics 封装 e2e） | 65.4ms | 含前后处理 |
| 对照：ONNX Runtime FP32（实验 001） | 20.7ms | 同模型 |
| 对照：OpenVINO INT8（实验 003） | 33.0ms | 同精度口径见实验 003 |

## 观察

- **NCNN 在 x86 笔记本上不占优**：NCNN 的优化主战场是 ARM 移动端（NEON 汇编 kernel），在 Intel CPU 上明显慢于 OpenVINO/ONNX Runtime——这本身就是部署选型的重要一课：**引擎与硬件的匹配比引擎名气重要**。
- 手写管线与官方封装结果一致，证明前处理/解码/NMS 的实现没有精度损失——这套逻辑可原样移植到 C++/Android。
- ultralytics 封装的 e2e（65ms）比纯推理（48ms）慢 17ms，差值主要是其 Python 前后处理开销——端侧部署中用原生代码写前后处理的价值就在这里。

## 踩坑

- `ncnn.Mat(numpy_hwc_uint8)` 直接构造会**段错误**（pyncnn 对非预期 layout 无报错直接崩溃）；正确做法是 `ncnn.Mat.from_pixels(rgb, PIXEL_RGB, w, h)`。
- NCNN 导出产物：`model.ncnn.param`（计算图文本，可用 Netron 可视化）+ `model.ncnn.bin`（权重）。

## 复现

```bash
python scripts/ncnn_manual_bench.py
```

原始数据见 [008-ncnn-manual.json](008-ncnn-manual.json)。

## 后续

- 安装 C++ 工具链（MSYS2/MinGW）后将本管线移植为原生 C++，测原生 NCNN 性能
- 手机端：Android + NCNN 是同一代码逻辑的最终归宿（零成本真机端侧验证）
