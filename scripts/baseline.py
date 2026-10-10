"""基线实验：跑通 YOLO 推理并记录 模型大小 / 推理延迟 指标。

用法:
    python scripts/baseline.py                 # 默认测 yolo26n 和 yolov8n
    python scripts/baseline.py --models yolo26n.pt
"""

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
TEST_IMAGE = "https://ultralytics.com/images/bus.jpg"

WARMUP = 3
REPEAT = 10


def measure(model: YOLO) -> dict:
    for _ in range(WARMUP):
        model.predict(TEST_IMAGE, verbose=False)

    total_ms, infer_ms, pre_ms, post_ms = [], [], [], []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        result = model.predict(TEST_IMAGE, verbose=False)[0]
        total_ms.append((time.perf_counter() - t0) * 1000)
        speed = result.speed
        pre_ms.append(speed["preprocess"])
        infer_ms.append(speed["inference"])
        post_ms.append(speed["postprocess"])

    avg = lambda xs: round(sum(xs) / len(xs), 2)
    return {
        "e2e_ms": avg(total_ms),
        "preprocess_ms": avg(pre_ms),
        "inference_ms": avg(infer_ms),
        "postprocess_ms": avg(post_ms),
    }


def run_one(weights: str) -> dict:
    model = YOLO(weights)
    record = {
        "model": weights,
        "pt_size_mb": round(Path(weights).stat().st_size / 1024**2, 2),
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "latency": measure(model),
    }

    onnx_path = Path(model.export(format="onnx", imgsz=640, verbose=False))
    record["onnx_size_mb"] = round(onnx_path.stat().st_size / 1024**2, 2)

    onnx_model = YOLO(str(onnx_path), task="detect")
    record["latency_onnx"] = measure(onnx_model)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["yolo26n.pt", "yolov8n.pt"])
    args = parser.parse_args()

    BENCH_DIR.mkdir(exist_ok=True)
    records = [run_one(w) for w in args.models]

    env = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "os": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "ultralytics": ultralytics.__version__,
        "imgsz": 640,
        "warmup": WARMUP,
        "repeat": REPEAT,
    }
    result = {"env": env, "records": records}

    out_json = BENCH_DIR / "001-baseline.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False))

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
    print(f"\n结果已保存: {out_json}")


if __name__ == "__main__":
    main()
