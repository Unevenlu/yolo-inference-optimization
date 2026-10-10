# 001 - 基线：跑通 YOLO 推理

**日期**: 2026-10-10

## 目的

跑通 Ultralytics 基础流程（下载权重 → 推理 → 导出 ONNX → ONNX Runtime 推理），建立后续所有优化实验的对照基线。

## 环境

| 项目 | 值 |
|------|-----|
| CPU | 12th Gen Intel Core i7-1260P |
| OS | Windows 11 (10.0.26100) |
| Python | 3.11.17 (conda env: yolo) |
| torch | 2.14.1+cpu |
| ultralytics | 8.4.175 |
| onnxruntime | 1.31.0 (CPUExecutionProvider) |
| 输入尺寸 | 640×640 |
| 测速方式 | warmup 3 次，正式 10 次取均值，测试图 bus.jpg |

## 结果

| 模型 | 参数量(M) | GFLOPs | .pt 大小(MB) | .onnx 大小(MB) | PyTorch 推理(ms) | ONNX 推理(ms) | 加速比 |
|------|-----------|--------|--------------|----------------|------------------|---------------|--------|
| yolo26n | 2.57 | 5.5 | 5.29 | 9.47 | 38.57 | 20.74 | 1.86x |
| yolov8n | 3.16 | 8.7 | 6.25 | 12.26 | 38.47 | 28.55 | 1.35x |

> 注：上表"推理"为纯 inference 耗时。端到端（含前后处理）：yolo26n PyTorch 48.55ms → ONNX 36.91ms；yolov8n 46.88ms → 45.03ms。

## 观察

- **yolo26n 更轻**：参数量比 yolov8n 少 19%，FLOPs 少 37%。
- **yolo26n 在 ONNX Runtime 上明显更快**（20.74 vs 28.55ms，快 27%），与其面向端侧的架构设计（DFL-free 检测头）一致；而在 PyTorch 下两者几乎相同，说明 26n 的优势要在推理引擎中才能体现。
- 导出 ONNX 后文件变大是正常的：`.pt` 只存权重，`.onnx` 同时存储计算图与权重常量。
- yolo26n 后处理仅 ~1ms（NMS-free 端到端输出），v8n 后处理 0.88-2.3ms（含 NMS）。

## 复现

```bash
python scripts/baseline.py                 # 默认两个模型都测
python scripts/baseline.py --models yolo26n.pt
```

原始数据见 [001-baseline.json](001-baseline.json)。
