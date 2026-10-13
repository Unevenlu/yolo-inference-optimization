"""实验 004：结构化剪枝 + 微调恢复精度。

流程（全部在单个进程内完成，避免剪枝模型存取的结构兼容问题）:
    1. 基线: yolov8n 在 32 张留出验证图上的 mAP / 参数量 / FLOPs / 延迟
    2. 剪枝: torch-pruning 按通道 L2 重要性剪掉 30% 通道（保护检测头）
    3. 剪枝后直接测精度 —— 预期明显下降
    4. 微调: 用 96 张训练图（与验证集不重叠）小学习率训练若干 epoch
    5. 微调后再测精度 —— 观察恢复程度

数据纪律: coco128 切成 96 训练 / 32 验证两个互斥子集，微调绝不接触验证图。

剪枝方案参考 Torch-Pruning 官方 yolov8 示例:
https://github.com/VainF/Torch-Pruning/blob/master/examples/yolov8/yolov8_pruning.py
两个关键点均来自该示例:
    a. C2f 的 chunk 操作依赖图无法追踪，必须替换为等价的 C2f_v2（cv1 拆成 cv0/cv1）
    b. 整个 Detect 头加入 ignored_layers

用法:
    python scripts/prune_finetune_bench.py
"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck

import torch_pruning as tp

from bench_utils import measure, collect_env, save_result, size_mb

WEIGHTS = "yolov8n.pt"
CH_SPARSITY = 0.3        # 剪掉 30% 通道
FINETUNE_EPOCHS = 10     # 与官方示例一致；剪枝后模型近乎失能，需要足够 epoch 恢复
IMGSZ = 640

COCO128_DIR = Path("datasets/coco128")
SPLIT_YAML = Path("datasets/coco128_split.yaml")


# ---------- C2f 替换（依赖图无法追踪 chunk，拆成两个独立卷积，行为等价） ----------

class C2f_v2(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n))

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def _transfer_weights(c2f: C2f, c2f_v2: C2f_v2) -> None:
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m
    sd, sd_v2 = c2f.state_dict(), c2f_v2.state_dict()
    half = sd["cv1.conv.weight"].shape[0] // 2
    sd_v2["cv0.conv.weight"] = sd["cv1.conv.weight"][:half]
    sd_v2["cv1.conv.weight"] = sd["cv1.conv.weight"][half:]
    for k in ["weight", "bias", "running_mean", "running_var"]:
        sd_v2[f"cv0.bn.{k}"] = sd[f"cv1.bn.{k}"][:half]
        sd_v2[f"cv1.bn.{k}"] = sd[f"cv1.bn.{k}"][half:]
    for k in sd:
        if not k.startswith("cv1."):
            sd_v2[k] = sd[k]
    c2f_v2.load_state_dict(sd_v2)
    # 复制 ultralytics 的模块元数据（f/i/type/np 等），_predict_once 依赖它们
    for attr in dir(c2f):
        v = getattr(c2f, attr)
        if not callable(v) and "_" not in attr:
            setattr(c2f_v2, attr, v)


def replace_c2f(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, C2f) and not isinstance(child, C2f_v2):
            b = child.m[0]
            shortcut = (b.cv1.conv.in_channels == b.cv2.conv.out_channels
                        and hasattr(b, "add") and b.add)
            c2f_v2 = C2f_v2(child.cv1.conv.in_channels, child.cv2.conv.out_channels,
                            n=len(child.m), shortcut=shortcut,
                            g=child.m[0].cv2.conv.groups,
                            e=child.c / child.cv2.conv.out_channels)
            _transfer_weights(child, c2f_v2)
            setattr(module, name, c2f_v2)
        else:
            replace_c2f(child)


# ---------- 数据拆分 ----------

def make_split_yaml() -> Path:
    """把 coco128 拆成 96 训练 / 32 验证，返回新 yaml 路径。"""
    images = sorted((COCO128_DIR / "images/train2017").glob("*.jpg"))
    assert len(images) == 128, f"coco128 图片数量异常: {len(images)}"
    split_dir = COCO128_DIR / "splits"
    split_dir.mkdir(exist_ok=True)
    (split_dir / "train.txt").write_text(
        "\n".join(str(p.resolve()) for p in images[:96]))
    (split_dir / "val.txt").write_text(
        "\n".join(str(p.resolve()) for p in images[96:]))

    import ultralytics
    pkg_yaml = Path(ultralytics.__file__).parent / "cfg/datasets/coco128.yaml"
    src = yaml.safe_load(pkg_yaml.read_text(encoding="utf-8"))

    cfg = {
        "path": str(COCO128_DIR.resolve()),
        "train": "splits/train.txt",
        "val": "splits/val.txt",
        "names": src["names"],
    }
    SPLIT_YAML.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return SPLIT_YAML


# ---------- 指标采集 ----------

def stats_of(model: YOLO) -> dict:
    """参数量与 GFLOPs（thop 口径，与 ultralytics 报告一致）。"""
    flops, params = 0.0, 0.0
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params
        flops = get_flops(model.model, imgsz=IMGSZ)
        params = get_num_params(model.model) / 1e6
    except Exception as e:
        print(f"  stats 统计失败: {e}")
    return {"params_m": round(params, 2), "gflops": round(flops, 2)}


def weights_size_mb(model: YOLO) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = Path(f.name)
    torch.save(model.model.state_dict(), tmp)
    size = size_mb(tmp)
    tmp.unlink()
    return size


def evaluate(model: YOLO) -> dict:
    res = model.val(data=str(SPLIT_YAML), imgsz=IMGSZ, verbose=False)
    return {"map50_95": round(res.box.map, 4), "map50": round(res.box.map50, 4)}


def snapshot(stage: str, model: YOLO) -> dict:
    rec = {"stage": stage, "size_mb": weights_size_mb(model)}
    rec.update(stats_of(model))
    rec["latency"] = measure(model)
    rec["accuracy"] = evaluate(model)
    print(f"  [{stage}] params={rec['params_m']}M gflops={rec['gflops']} "
          f"size={rec['size_mb']}MB mAP50-95={rec['accuracy']['map50_95']}")
    return rec


# ---------- 剪枝 ----------

def prune(model: YOLO) -> None:
    """对 yolov8n 做结构化剪枝；整个 Detect 头受保护（输出通道数有语义约束）。"""
    net = model.model
    replace_c2f(net)

    # ultralytics 加载的模型 requires_grad=False（推理态），
    # torch-pruning 依赖 autograd 构建依赖图，必须打开梯度。
    # 保持 eval 模式：避免追踪时的随机输入污染 BN 统计量
    net.eval()
    for p in net.parameters():
        p.requires_grad_(True)

    example = torch.randn(1, 3, IMGSZ, IMGSZ)
    ignored = [m for m in net.modules() if isinstance(m, Detect)]

    pruner = tp.pruner.MetaPruner(
        net, example,
        importance=tp.importance.MagnitudeImportance(p=2),   # L2 范数重要性
        pruning_ratio=CH_SPARSITY,
        ignored_layers=ignored,
    )
    pruner.step()
    net.eval()


def main() -> None:
    make_split_yaml()
    records = []

    model = YOLO(WEIGHTS)
    records.append(snapshot("baseline", model))

    print(f"\n剪枝中 (pruning_ratio={CH_SPARSITY}) ...")
    prune(model)
    records.append(snapshot("pruned", model))

    print(f"\n微调中 (epochs={FINETUNE_EPOCHS}, 96 张训练图) ...")
    # 两个关键细节：
    # 1. 不能用 model.train()——它会按 yaml 重建模型并加载原始权重，丢弃剪枝结构。
    #    必须手动构造 trainer 后直接替换其 model。
    # 2. 必须显式指定 SGD——默认 optimizer='auto' 会改用 AdamW 并忽略 lr0。
    trainer = DetectionTrainer(overrides={
        "model": WEIGHTS, "data": str(SPLIT_YAML), "epochs": FINETUNE_EPOCHS,
        "imgsz": IMGSZ, "batch": 16, "optimizer": "SGD", "lr0": 0.01,
        "momentum": 0.937, "warmup_epochs": 1.0, "cos_lr": True,
        "workers": 0, "save": False, "val": False, "plots": False,
        "verbose": False,
    })
    trainer.model = model.model
    trainer.train()
    model.model = trainer.model
    records.append(snapshot("pruned+finetuned", model))

    # 归档微调后权重（.pt 已被 gitignore；含自定义 C2f_v2，需本脚本环境才能加载）
    torch.save({"model": model.model}, "yolov8n_pruned_finetuned.pt")

    env = collect_env()
    env.update({
        "val_data": "coco128 32 张留出子集",
        "train_data": "coco128 96 张子集",
        "ch_sparsity": CH_SPARSITY,
        "finetune_epochs": FINETUNE_EPOCHS,
        "note": "微调与验证使用互斥子集；mAP 与实验 003 的全量 coco128 口径不同，"
                "只能在本实验内部纵向比较",
    })
    out = save_result("004-prune-finetune", env, records)

    header = f"{'stage':<20} {'params(M)':>10} {'GFLOPs':>8} {'size(MB)':>9} " \
             f"{'infer(ms)':>10} {'mAP50-95':>9} {'mAP50':>7}"
    print("\n" + header)
    print("-" * len(header))
    for r in records:
        print(f"{r['stage']:<20} {r['params_m']:>10} {r['gflops']:>8} "
              f"{r['size_mb']:>9} {r['latency']['inference_ms']:>10} "
              f"{r['accuracy']['map50_95']:>9} {r['accuracy']['map50']:>7}")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
