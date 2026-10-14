"""实验 006：迭代式剪枝——验证实验 004/005 的核心结论。

实验 004/005 证明：一次性剪枝 30% 使模型崩溃，恢复成本≈重新训练；
生产配方是"小步多次，每步微调，永不崩溃"。本实验直接验证：

    3 轮迭代 × 每轮剪 11.2%（累计≈30%，与一次性 30% 等比）
    每轮: 剪枝 → 测 mAP（预期下降但不为零）→ 1000 图微调 10 epoch → 测 mAP（预期恢复）

对照（同一 500 张留出验证集）:
    - 一次性 30%: 剪后 0.0，恢复后 0.0279（实验 005）
    - 迭代式 30%: 见本实验

用法:
    python scripts/iterative_prune_bench.py
"""

import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prune_finetune_bench as pfb  # noqa: E402
from prune_finetune_bench import prune  # noqa: E402
from prune_recovery_coco import ensure_data, evaluate, finetune  # noqa: E402
from bench_utils import measure, collect_env, save_result  # noqa: E402

WEIGHTS = "yolov8n.pt"
TARGET_TOTAL = 0.30          # 累计目标剪枝率（与实验 004/005 等比）
ITERATIONS = 3
PER_STEP = 1 - (1 - TARGET_TOTAL) ** (1 / ITERATIONS)   # ≈ 0.112


def snapshot(stage: str, model: YOLO) -> dict:
    rec = {"stage": stage}
    rec.update(pfb.stats_of(model))
    rec["accuracy"] = evaluate(model)
    print(f"  [{stage}] params={rec['params_m']}M gflops={rec['gflops']} "
          f"mAP50-95={rec['accuracy']['map50_95']}")
    return rec


def main() -> None:
    ensure_data()
    print(f"迭代方案: {ITERATIONS} 轮 × 每轮 {PER_STEP:.1%}（累计≈{TARGET_TOTAL:.0%}）")
    records = []

    model = YOLO(WEIGHTS)
    records.append(snapshot("baseline", model))

    for i in range(ITERATIONS):
        print(f"\n=== 第 {i + 1}/{ITERATIONS} 轮 ===")
        pfb.CH_SPARSITY = PER_STEP
        prune(model)
        records.append(snapshot(f"iter{i+1}-pruned", model))

        print(f"  微调中 (10 epochs × 1000 张) ...")
        finetune(model)
        records.append(snapshot(f"iter{i+1}-recovered", model))

    # 终态：补延迟与大小
    final = records[-1]
    final["latency"] = measure(model)
    final["size_mb"] = pfb.weights_size_mb(model)
    torch.save({"model": model.model}, "yolov8n_iterative_pruned.pt")

    env = collect_env()
    env.update({
        "data": "COCO val2017 子集: 1000 训练 / 500 验证（互斥）",
        "iterations": ITERATIONS,
        "per_step_ratio": round(PER_STEP, 4),
        "target_total": TARGET_TOTAL,
        "finetune_epochs_per_step": 10,
    })
    out = save_result("006-iterative-prune", env, records)

    print(f"\n{'stage':<18} {'params(M)':>10} {'GFLOPs':>8} {'mAP50-95':>9} {'mAP50':>7}")
    for r in records:
        print(f"{r['stage']:<18} {r['params_m']:>10} {r['gflops']:>8} "
              f"{r['accuracy']['map50_95']:>9} {r['accuracy']['map50']:>7}")
    print(f"\n终态延迟: {final['latency']['inference_ms']}ms  大小: {final['size_mb']}MB")
    print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()
