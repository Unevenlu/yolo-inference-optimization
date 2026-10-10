# 002 - OpenVINO：Intel CPU 专用推理引擎

**日期**: 2026-10-10

## 目的

在 Intel i7-1260P 上对比 yolo26n 的三种部署形态：ONNX Runtime（FP32）、OpenVINO FP32、OpenVINO FP16，验证 OpenVINO 在 Intel 硬件上的收益。

## 环境

同实验 001，新增 openvino 2026.4.1。所有引擎在**同一次运行内**交替测量，排除机器状态差异。

## 结果（稳定运行）

| 引擎 | 大小(MB) | e2e(ms) | 前处理(ms) | 推理(ms) | 后处理(ms) |
|------|----------|---------|------------|----------|------------|
| ONNX Runtime (FP32) | 9.47 | 34.03 | 3.93 | 20.79 | 1.46 |
| OpenVINO (FP32) | 9.64 | 31.42 | 2.22 | 23.14 | 1.06 |
| OpenVINO (FP16) | **5.12** | **28.79** | 2.36 | **20.28** | 0.97 |

对照实验 001：PyTorch 推理 38.57ms。OpenVINO FP16 端到端相比 PyTorch 基线提速约 **1.9x**，模型体积缩小约一半（相对 .pt 的 5.29MB 也更小）。

## 观察

- **FP16 是最划算的一步**：体积减半（9.64 → 5.12MB），推理还略快于 FP32。CPU 上 FP16 的收益主要来自内存带宽减半（i7-1260P 的 AVX2 不原生加速 FP16 计算，但 OpenVINO 会转换优化）。
- **OpenVINO FP32 的纯推理并没有赢过 ONNX Runtime**（23.14 vs 20.79ms），但前后处理开销明显更低（合计 3.3ms vs 5.4ms），端到端反超。引擎比较要看 e2e，不能只看 inference。
- **测量稳定性教训**：本实验第一次运行恰逢系统高负载，三个引擎全部慢了 ~75%（ONNX 甚至测出 36.8ms），但**相对关系不变**。机器安静时复测，ONNX 结果 20.79ms 与实验 001 的 20.74ms 几乎完全一致——同场对比 + 复测是可信数据的基本保障。

## 踩坑

- Ultralytics 靠目录名后缀 `_openvino_model` 识别 OpenVINO 模型，**导出目录改名会导致无法加载**（报 "not a supported model format"）。FP32/FP16 两次导出只能先后覆盖同一目录，大小时刻分别记录。
- `half=True` 参数已被标记废弃，官方建议改用 `quantize`（INT8 量化）——正是实验 003 的方向。

## 待补

- 本实验只关注了大小与延迟，FP16 的精度损失尚未量化。将在量化实验中用 COCO val 子集统一补测 mAP。

## 复现

```bash
python scripts/openvino_bench.py
```

原始数据见 [002-openvino.json](002-openvino.json)。
