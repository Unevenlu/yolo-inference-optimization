"""基线实验：跑通 YOLO 推理并记录 模型大小 / 推理延迟 指标。

用法:
    python scripts/baseline.py                 # 默认测 yolo26n 和 yolov8n
    python scripts/baseline.py --models yolo26n.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from bench_utils import measure, collect_env, save_result, size_mb, TEST_IMAGE  # noqa: E402


def run_one(weights: str) -> dict:
    model = YOLO(weights)
    record = {
        "model": weights,
        "pt_size_mb": size_mb(weights),
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "latency": measure(model),
    }

    onnx_path = Path(model.export(format="onnx", imgsz=640, verbose=False))
    record["onnx_size_mb"] = size_mb(onnx_path)

    onnx_model = YOLO(str(onnx_path), task="detect")
    record["latency_onnx"] = measure(onnx_model)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["yolo26n.pt", "yolov8n.pt"])
    args = parser.parse_args()

    records = [run_one(w) for w in args.models]
    env = collect_env()
    out = save_result("001-baseline", env, records)

    print(f"\n环境: Python {env['python']} | torch {env['torch']} | "
          f"ultralytics {env['ultralytics']} | CUDA {env['cuda_available']}")
    header = f"{'model':<12} {'pt(MB)':>7} {'onnx(MB)':>9} {'params(M)':>9} " \
             f"{'e2e(ms)':>8} {'infer(ms)':>9} {'onnx e2e':>9} {'onnx infer':>10}"
    print(header)
    print("-" * len(header))
    for r in records:
        print(f"{r['model']:<12} {r['pt_size_mb']:>7} {r['onnx_size_mb']:>9} "
              f"{r['params_m']:>9} {r['latency']['e2e_ms']:>8} "
              f"{r['latency']['inference_ms']:>9} {r['latency_onnx']['e2e_ms']:>9} "
              f"{r['latency_onnx']['inference_ms']:>10}")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
