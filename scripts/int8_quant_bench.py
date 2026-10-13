"""实验 003：INT8 量化（NNCF）——大小 / 延迟 / 精度三者权衡。

对比 yolo26n 的三种精度形态:
    1. FP32 (PyTorch .pt)      —— 精度基准
    2. FP16 (OpenVINO)         —— 实验 002 的产物
    3. INT8 (OpenVINO + NNCF)  —— 本实验目标

与之前实验不同，量化可能损失精度，因此引入 mAP 测量。
验证集用 coco128（COCO 的 128 张图子集），INT8 校准也用同一数据集。

用法:
    python scripts/int8_quant_bench.py
"""

from pathlib import Path

from ultralytics import YOLO

from bench_utils import measure, collect_env, save_result, size_mb

WEIGHTS = "yolo26n.pt"
FP16_DIR = Path("yolo26n_openvino_model")       # 实验 002 的 FP16 产物
DATA = "coco128.yaml"


def evaluate(yolo: YOLO) -> dict:
    """返回 COCO val 指标（mAP50-95 / mAP50）。"""
    res = yolo.val(data=DATA, imgsz=640, verbose=False)
    return {"map50_95": round(res.box.map, 4), "map50": round(res.box.map50, 4)}


def record_one(name: str, artifact: str, yolo: YOLO) -> dict:
    return {
        "engine": name,
        "artifact": artifact,
        "size_mb": size_mb(artifact),
        "latency": measure(yolo),
        "accuracy": evaluate(yolo),
    }


def main() -> None:
    records = []

    # 1. FP32 PyTorch 基准
    model = YOLO(WEIGHTS)
    records.append(record_one("PyTorch (FP32)", WEIGHTS, model))

    # 2. OpenVINO FP16（实验 002 产物；不存在则重新导出）
    if not FP16_DIR.exists():
        model.export(format="openvino", imgsz=640, half=True, verbose=False)
    records.append(record_one("OpenVINO (FP16)", str(FP16_DIR),
                              YOLO(str(FP16_DIR), task="detect")))

    # 3. OpenVINO INT8（NNCF 量化，校准数据用 coco128）
    int8_dir = Path(model.export(format="openvino", imgsz=640, int8=True,
                                 data=DATA, verbose=False))
    records.append(record_one("OpenVINO (INT8)", str(int8_dir),
                              YOLO(str(int8_dir), task="detect")))

    env = collect_env()
    env["val_data"] = DATA
    out = save_result("003-int8-quant", env, records)

    header = f"{'engine':<18} {'size(MB)':>9} {'infer(ms)':>10} {'e2e(ms)':>9} " \
             f"{'mAP50-95':>9} {'mAP50':>7}"
    print("\n" + header)
    print("-" * len(header))
    for r in records:
        lat, acc = r["latency"], r["accuracy"]
        print(f"{r['engine']:<18} {r['size_mb']:>9} {lat['inference_ms']:>10} "
              f"{lat['e2e_ms']:>9} {acc['map50_95']:>9} {acc['map50']:>7}")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
