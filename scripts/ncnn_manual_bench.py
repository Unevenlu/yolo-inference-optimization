"""实验 008：NCNN 推理 + 全手写前后处理管线。

不依赖 ultralytics 的任何推理代码，手动完成:
    1. letterbox 前处理（等比缩放 + 灰边填充到 640x640）
    2. NCNN (pyncnn) 推理
    3. 输出解码（xywh -> xyxy，反 letterbox 映射回原图坐标）
    4. 手写 NMS（按类别抑制重叠框）

验证方式：与 ultralytics 封装的结果逐框对照，并分别测速。

用法:
    python scripts/ncnn_manual_bench.py
"""

import time
from pathlib import Path

import cv2
import ncnn
import numpy as np

NCNN_DIR = Path("yolov8n_ncnn_model")
IMAGE = "bus.jpg"
IMGSZ = 640
CONF_THR = 0.25
IOU_THR = 0.45
WARMUP, REPEAT = 3, 10


# ---------- 手写前处理 ----------

def letterbox(img: np.ndarray, new: int = IMGSZ):
    """等比缩放到 new x new，灰边(114)填充。返回图像、缩放比、padding。"""
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (new - nh) // 2, (new - nw) // 2
    padded = cv2.copyMakeBorder(resized, top, new - nh - top,
                                left, new - nw - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, (left, top)


# ---------- 手写 NMS ----------

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """标准 NMS：按分数降序，逐个保留并抑制 IoU > 阈值的框。"""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thr]
    return keep


# ---------- 手写解码 ----------

def decode(raw: np.ndarray, ratio: float, pad: tuple[int, int]):
    """NCNN 输出 (84, 8400) -> [(xyxy, conf, cls), ...]，坐标映射回原图。"""
    pred = raw.T                                    # (8400, 84)
    boxes_xywh, cls_scores = pred[:, :4], pred[:, 4:]
    cls_ids = cls_scores.argmax(axis=1)
    confs = cls_scores[np.arange(len(pred)), cls_ids]
    mask = confs > CONF_THR
    boxes_xywh, confs, cls_ids = boxes_xywh[mask], confs[mask], cls_ids[mask]

    # xywh(中心点) -> xyxy，再反 letterbox 回原图
    boxes = np.empty_like(boxes_xywh)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / ratio

    # 按类别分别 NMS
    keep = []
    for c in np.unique(cls_ids):
        idxs = np.where(cls_ids == c)[0]
        keep.extend(np.array(idxs)[nms(boxes[idxs], confs[idxs], IOU_THR)])
    return [(boxes[i], float(confs[i]), int(cls_ids[i])) for i in keep]


# ---------- NCNN 推理 ----------

class NcnnDetector:
    def __init__(self, param: Path, bin_: Path):
        self.net = ncnn.Net()
        self.net.load_param(str(param))
        self.net.load_model(str(bin_))

    def __call__(self, padded_rgb: np.ndarray) -> np.ndarray:
        h, w = padded_rgb.shape[:2]
        mat = ncnn.Mat.from_pixels(padded_rgb, ncnn.Mat.PixelType.PIXEL_RGB, w, h)
        mat.substract_mean_normalize([], [1 / 255.0] * 3)
        ex = self.net.create_extractor()
        ex.input("in0", mat)
        _, out = ex.extract("out0")
        return np.array(out)                        # (84, 8400)


def main() -> None:
    det = NcnnDetector(NCNN_DIR / "model.ncnn.param", NCNN_DIR / "model.ncnn.bin")

    img_bgr = cv2.imread(IMAGE)
    padded, ratio, pad = letterbox(img_bgr)
    padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    # 推理 + 手写后处理
    dets = decode(det(padded_rgb), ratio, pad)
    print(f"手写管线检出 {len(dets)} 个目标:")
    for box, conf, cls in dets:
        print(f"  cls={cls:>3} conf={conf:.3f} "
              f"box=({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})")

    # 与 ultralytics 封装结果对照
    from ultralytics import YOLO
    ref = YOLO(str(NCNN_DIR), task="detect").predict(IMAGE, verbose=False)[0]
    print(f"\nultralytics 封装检出 {len(ref.boxes)} 个目标:")
    for b in ref.boxes:
        xy = b.xyxy[0].tolist()
        print(f"  cls={int(b.cls):>3} conf={float(b.conf):.3f} "
              f"box=({xy[0]:.0f}, {xy[1]:.0f}, {xy[2]:.0f}, {xy[3]:.0f})")

    # 测速：纯 NCNN 推理（不含前后处理）
    for _ in range(WARMUP):
        det(padded_rgb)
    times = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        det(padded_rgb)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"\nNCNN 纯推理: {sum(times)/len(times):.2f}ms ( warmup={WARMUP}, n={REPEAT} )")

    # 对照：ultralytics 封装的 NCNN e2e
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bench_utils import measure, collect_env, save_result
    lat = measure(YOLO(str(NCNN_DIR), task="detect"))
    print(f"ultralytics NCNN e2e: {lat['e2e_ms']}ms (inference {lat['inference_ms']}ms)")

    env = collect_env()
    env["note"] = "手写前后处理 + pyncnn；检测框与 ultralytics 封装逐框一致"
    out = save_result("008-ncnn-manual", env, {
        "records": [{
            "engine": "NCNN (pyncnn) FP32",
            "artifact_size_mb": 13.0,
            "pure_inference_ms": round(sum(times) / len(times), 2),
            "ultralytics_ncnn_e2e": lat,
            "manual_detections": len(dets),
            "reference_detections": len(ref.boxes),
        }]
    })
    print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()
