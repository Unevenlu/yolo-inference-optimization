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

## 环境

- CPU: 12th Gen Intel Core i7-1260P
- OS: Windows 11
- Python: 3.11.17（conda 环境 `yolo`）
- torch 2.14.1+cpu / ultralytics 8.4.175 / onnxruntime 1.31.0

## 快速开始

```bash
conda activate yolo
python scripts/baseline.py
```
