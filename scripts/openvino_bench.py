"""实验 002：OpenVINO 推理（Intel CPU 专用引擎）。

对比 yolo26n 在同一次运行中的三种形态:
    1. ONNX Runtime  (实验 001 的对照，同场复测保证可比性)
    2. OpenVINO FP32
    3. OpenVINO FP16 (半精度)

注意: Ultralytics 通过目录名后缀 `_openvino_model` 识别 OpenVINO 模型，
目录不能改名，因此 FP32 测完先记录大小，再导出 FP16 覆盖同一目录。

用法:
    python scripts/openvino_bench.py
"""

from pathlib import Path

from ultralytics import YOLO

from bench_utils import measure, collect_env, save_result, size_mb

WEIGHTS = "yolo26n.pt"


def main() -> None:
    model = YOLO(WEIGHTS)
    records = []

    # 1. ONNX 对照（复用实验 001 的导出产物）
    onnx_path = Path(WEIGHTS).with_suffix(".onnx")
    onnx_model = YOLO(str(onnx_path), task="detect")
    records.append({
        "engine": "ONNX Runtime (FP32)",
        "artifact": onnx_path.name,
        "size_mb": size_mb(onnx_path),
        "latency": measure(onnx_model),
    })

    # 2. OpenVINO FP32
    ov_dir = Path(model.export(format="openvino", imgsz=640, verbose=False))
    fp32_size = size_mb(ov_dir)
    records.append({
        "engine": "OpenVINO (FP32)",
        "artifact": ov_dir.name,
        "size_mb": fp32_size,
        "latency": measure(YOLO(str(ov_dir), task="detect")),
    })

    # 3. OpenVINO FP16（覆盖同一目录，FP32 大小已在上面记录）
    ov_dir = Path(model.export(format="openvino", imgsz=640, half=True, verbose=False))
    records.append({
        "engine": "OpenVINO (FP16)",
        "artifact": ov_dir.name,
        "size_mb": size_mb(ov_dir),
        "latency": measure(YOLO(str(ov_dir), task="detect")),
    })

    env = collect_env()
    out = save_result("002-openvino", env, records)

    header = f"{'engine':<22} {'size(MB)':>9} {'e2e(ms)':>9} {'pre(ms)':>8} " \
             f"{'infer(ms)':>10} {'post(ms)':>9}"
    print("\n" + header)
    print("-" * len(header))
    for r in records:
        lat = r["latency"]
        print(f"{r['engine']:<22} {r['size_mb']:>9} {lat['e2e_ms']:>9} "
              f"{lat['preprocess_ms']:>8} {lat['inference_ms']:>10} "
              f"{lat['postprocess_ms']:>9}")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
