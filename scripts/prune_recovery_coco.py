"""实验 005：真实数据量下的剪枝恢复（COCO val2017 子集）。

实验 004 证明：一次性剪枝 30% 使 yolov8n 崩溃，96 张图微调无法恢复。
本实验验证核心假设：**恢复失败的关键变量是数据量**。

设计:
    - 数据: COCO val2017 切分 1000 训练 / 500 留出验证（互斥）
    - 基线: 未剪枝 yolov8n 在 500 张验证图上的 mAP（可信口径）
    - 剪枝: 一次性 30%（同实验 004）
    - 恢复: 1000 张训练图，SGD(lr0=0.01) 10 epoch（约 620 步）

用法:
    python scripts/prune_recovery_coco.py
"""

import sys
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prune_finetune_bench import prune, replace_c2f  # noqa: E402
from bench_utils import measure, collect_env, save_result, size_mb  # noqa: E402

WEIGHTS = "yolov8n.pt"
IMGSZ = 640
N_TRAIN, N_VAL = 1000, 500
EPOCHS = 10

COCO_DIR = Path("datasets/coco")
SPLIT_YAML = Path("datasets/coco/coco_subset.yaml")


def ensure_data() -> None:
    """校验数据并构建 1000/500 拆分 yaml。

    数据准备（网络受限时不整包下载）：
    标注来自 coco2017labels.zip，图片按标注 id 从
    http://images.cocodataset.org/val2017/{id}.jpg 并行单张下载前 1500 张。
    """
    images = sorted((COCO_DIR / "images/val2017").glob("*.jpg"))
    labels = list((COCO_DIR / "labels/val2017").glob("*.txt"))
    assert len(images) >= N_TRAIN + N_VAL, f"图片不足: {len(images)}"
    assert labels, "标注缺失"
    print(f"数据就绪: {len(images)} 张图, {len(labels)} 个标注")

    split_dir = COCO_DIR / "splits"
    split_dir.mkdir(exist_ok=True)
    (split_dir / "train1000.txt").write_text(
        "\n".join(str(p.resolve()) for p in images[:N_TRAIN]))
    (split_dir / "val500.txt").write_text(
        "\n".join(str(p.resolve()) for p in images[N_TRAIN:N_TRAIN + N_VAL]))

    import ultralytics
    pkg = Path(ultralytics.__file__).parent / "cfg/datasets/coco.yaml"
    src = yaml.safe_load(pkg.read_text(encoding="utf-8"))
    SPLIT_YAML.write_text(yaml.safe_dump({
        "path": str(COCO_DIR.resolve()),
        "train": "splits/train1000.txt",
        "val": "splits/val500.txt",
        "names": src["names"],
    }, allow_unicode=True), encoding="utf-8")


def evaluate(model: YOLO) -> dict:
    res = model.val(data=str(SPLIT_YAML), imgsz=IMGSZ, batch=16, verbose=False)
    return {"map50_95": round(res.box.map, 4), "map50": round(res.box.map50, 4)}


def finetune(model: YOLO) -> None:
    """显式 SGD（默认 auto 会改用 AdamW 并忽略 lr0，见实验 004）。"""
    trainer = DetectionTrainer(overrides={
        "model": WEIGHTS, "data": str(SPLIT_YAML), "epochs": EPOCHS,
        "imgsz": IMGSZ, "batch": 16, "optimizer": "SGD", "lr0": 0.01,
        "momentum": 0.937, "warmup_epochs": 1.0, "cos_lr": True,
        "workers": 0, "save": False, "val": False, "plots": False,
        "verbose": False,
    })
    trainer.model = model.model   # 关键：不让 trainer 重建模型
    trainer.train()
    model.model = trainer.model


def main() -> None:
    ensure_data()
    records = []

    model = YOLO(WEIGHTS)
    acc = evaluate(model)
    records.append({"stage": "baseline", "params_m": 3.16, "gflops": 8.86,
                    "accuracy": acc})
    print(f"[baseline] mAP50-95={acc['map50_95']}")

    print("\n剪枝中 (30%) ...")
    prune(model)
    acc = evaluate(model)
    records.append({"stage": "pruned", "params_m": 1.81, "gflops": 5.59,
                    "accuracy": acc})
    print(f"[pruned] mAP50-95={acc['map50_95']}")

    print(f"\n恢复训练中 ({EPOCHS} epochs × {N_TRAIN} 张, 约 50 分钟) ...")
    finetune(model)
    acc = evaluate(model)
    records.append({"stage": "pruned+recovered", "params_m": 1.81, "gflops": 5.59,
                    "accuracy": acc})
    print(f"[pruned+recovered] mAP50-95={acc['map50_95']}")

    import torch
    torch.save({"model": model.model}, "yolov8n_pruned_recovered.pt")

    env = collect_env()
    env.update({
        "data": f"COCO val2017 子集: {N_TRAIN} 训练 / {N_VAL} 验证（互斥）",
        "finetune_epochs": EPOCHS,
        "note": "与实验 004 的唯一变量是训练数据量（96 → 1000）与验证口径（32 → 500）",
    })
    out = save_result("005-prune-recovery-coco", env, records)

    print(f"\n{'stage':<20} {'mAP50-95':>9} {'mAP50':>7}")
    for r in records:
        print(f"{r['stage']:<20} {r['accuracy']['map50_95']:>9} "
              f"{r['accuracy']['map50']:>7}")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
