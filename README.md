# YOLO 推理优化实践记录

记录 YOLO 模型轻量化与端侧部署的学习过程：模型导出、推理引擎、量化、剪枝，以及每一步的 **模型大小 / 推理延迟 / 精度（mAP）** 三项指标对比。

## 目录结构

```
├── README.md
├── benchmarks/       # 每次实验的指标记录（大小 / 延迟 / 精度）
├── notebooks/        # 实验与分析 notebook
├── scripts/          # 训练、导出、推理脚本
└── docs/             # 学习笔记与踩坑记录
```

## 实验记录

| 日期 | 模型 | 优化手段 | 大小 | 延迟 | mAP | 备注 |
|------|------|----------|------|------|-----|------|
| 2026-10-10 | yolo26n | 基线 (PyTorch) | 5.29MB | 38.6ms | - | [报告](benchmarks/001-baseline.md) |
| 2026-10-10 | yolo26n | ONNX 导出 | 9.47MB | 20.7ms | - | 加速 1.86x |
| 2026-10-10 | yolov8n | 基线 (PyTorch) | 6.25MB | 38.5ms | - | 对照组 |
| 2026-10-10 | yolov8n | ONNX 导出 | 12.26MB | 28.6ms | - | 加速 1.35x |
| 2026-10-10 | yolo26n | OpenVINO FP32 | 9.64MB | 23.1ms | - | [报告](benchmarks/002-openvino.md) |
| 2026-10-10 | yolo26n | OpenVINO FP16 | 5.12MB | 20.3ms | - | 体积减半，e2e 较 PyTorch 提速 1.9x |
| 2026-10-13 | yolo26n | OpenVINO INT8 (NNCF) | 3.21MB | 33.0ms | 0.4685 | [报告](benchmarks/003-int8-quant.md)，mAP50-95 仅 -1.7 |
| 2026-10-13 | yolov8n | 剪枝 30% + 微调 | 7.06MB | 34.9ms | 0.0002 | [报告](benchmarks/004-prune-finetune.md)，一次性剪枝致崩溃，小数据无法恢复 |
| 2026-10-13 | yolov8n | 剪枝 30% + 1000图恢复 | 7.06MB | - | 0.0279 | [报告](benchmarks/005-prune-recovery-coco.md)，恢复成本≈重训，数据量提升140倍效果 |
| 2026-10-14 | yolov8n | 迭代剪枝 3×11.2% | 6.93MB | 42.7ms | 0.0734 | [报告](benchmarks/006-iterative-prune.md)，等比对照优于一次性 2.6 倍 |
| 2026-10-14 | yolov8n | 迭代剪枝 10×3.5% | 6.62MB | 36.3ms | 0.0683 | [报告](benchmarks/007-iterative-prune-10.md)，粒度不如总预算重要 |
| 2026-10-15 | yolov8n | NCNN + 手写前后处理 | 13MB | 48.2ms | - | [报告](benchmarks/008-ncnn-manual.md)，与官方封装逐框一致 |

## 环境

- CPU: 12th Gen Intel Core i7-1260P
- OS: Windows 11
- Python: 3.11.17（conda 环境 `yolo`）
- torch 2.14.1+cpu / ultralytics 8.4.175 / onnxruntime 1.31.0 / openvino 2026.4.1

## 快速开始

```bash
conda activate yolo
python scripts/baseline.py           # 实验 001: PyTorch / ONNX 基线
python scripts/openvino_bench.py     # 实验 002: OpenVINO FP32/FP16 对比
python scripts/int8_quant_bench.py   # 实验 003: INT8 量化（含 mAP 验证）
python scripts/prune_finetune_bench.py  # 实验 004: 结构化剪枝 + 微调恢复
python scripts/prune_recovery_coco.py   # 实验 005: 大数据量剪枝恢复（COCO 子集）
python scripts/iterative_prune_bench.py # 实验 006/007: 迭代式剪枝
python scripts/ncnn_manual_bench.py     # 实验 008: NCNN + 手写前后处理
```
