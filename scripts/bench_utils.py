"""实验共用的测速与记录工具。"""

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
    """预热 WARMUP 次后正式测 REPEAT 次，返回各项平均耗时(ms)。"""
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


def size_mb(path: Path) -> float:
    """文件或目录的总大小(MB)。"""
    path = Path(path)
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        total = path.stat().st_size
    return round(total / 1024**2, 2)


def collect_env() -> dict:
    return {
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


def save_result(name: str, env: dict, records: list) -> Path:
    BENCH_DIR.mkdir(exist_ok=True)
    out = BENCH_DIR / f"{name}.json"
    out.write_text(
        json.dumps({"env": env, "records": records}, indent=2, ensure_ascii=False)
    )
    return out


def print_table(records: list, columns: list) -> None:
    """columns: [(表头, 取值函数), ...]"""
    header = " ".join(f"{title:>12}" for title, _ in columns)
    print(header)
    print("-" * len(header))
    for r in records:
        print(" ".join(f"{str(get(r)):>12}" for _, get in columns))
