#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <net.h>

#include <algorithm>
#include <vector>

static ncnn::Net g_net;
static const int TARGET_SIZE = 640;
static const float CONF_THR = 0.25f;
static const float IOU_THR = 0.45f;

struct Det { float x1, y1, x2, y2, conf; int cls; };

// ---------- 手写 NMS（与实验 008 Python 版逻辑一致） ----------
static float iou(const Det& a, const Det& b) {
    float xx1 = std::max(a.x1, b.x1), yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2), yy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, xx2 - xx1) * std::max(0.f, yy2 - yy1);
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter);
}

static std::vector<Det> nms(std::vector<Det>& dets, float thr) {
    std::sort(dets.begin(), dets.end(),
              [](const Det& a, const Det& b) { return a.conf > b.conf; });
    std::vector<Det> keep;
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++)
            if (!suppressed[j] && dets[i].cls == dets[j].cls
                && iou(dets[i], dets[j]) > thr)
                suppressed[j] = true;
    }
    return keep;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_yoloncnn_app_YoloNcnn_loadModel(JNIEnv* env, jobject, jobject assetManager,
                                         jstring paramName, jstring binName) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    const char* param = env->GetStringUTFChars(paramName, nullptr);
    const char* bin = env->GetStringUTFChars(binName, nullptr);
    g_net.opt.num_threads = 4;
    int r1 = g_net.load_param(mgr, param);
    int r2 = g_net.load_model(mgr, bin);
    env->ReleaseStringUTFChars(paramName, param);
    env->ReleaseStringUTFChars(binName, bin);
    return (r1 == 0 && r2 == 0) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_yoloncnn_app_YoloNcnn_detect(JNIEnv* env, jobject, jobject bitmap) {
    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) return nullptr;

    void* pixels;
    AndroidBitmap_lockPixels(env, bitmap, &pixels);
    int srcW = info.width, srcH = info.height;

    // 前处理：letterbox（等比缩放 + 114 灰边填充）
    float scale = std::min((float) TARGET_SIZE / srcW, (float) TARGET_SIZE / srcH);
    int rw = (int) (srcW * scale), rh = (int) (srcH * scale);
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        (const unsigned char*) pixels, ncnn::Mat::PIXEL_RGBA2RGB,
        srcW, srcH, rw, rh);
    AndroidBitmap_unlockPixels(env, bitmap);

    int padX = (TARGET_SIZE - rw) / 2, padY = (TARGET_SIZE - rh) / 2;
    ncnn::Mat padded;
    ncnn::copy_make_border(in, padded, padY, TARGET_SIZE - rh - padY,
                           padX, TARGET_SIZE - rw - padX,
                           ncnn::BORDER_CONSTANT, 114.f);

    const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    padded.substract_mean_normalize(nullptr, norm);

    // 推理
    ncnn::Extractor ex = g_net.create_extractor();
    ex.input("in0", padded);
    ncnn::Mat out;                                // (84, 8400)
    ex.extract("out0", out);

    // 解码：xywh -> xyxy + 反 letterbox
    std::vector<Det> all;
    for (int j = 0; j < out.w; j++) {
        // 找最大类别分数
        float best = -1e9f;
        int bestC = 0;
        for (int c = 4; c < out.h; c++) {
            float s = out.row(c)[j];
            if (s > best) { best = s; bestC = c - 4; }
        }
        if (best < CONF_THR) continue;
        float cx = out.row(0)[j], cy = out.row(1)[j];
        float w = out.row(2)[j], h = out.row(3)[j];
        Det d;
        d.x1 = (cx - w / 2 - padX) / scale;
        d.y1 = (cy - h / 2 - padY) / scale;
        d.x2 = (cx + w / 2 - padX) / scale;
        d.y2 = (cy + h / 2 - padY) / scale;
        d.conf = best;
        d.cls = bestC;
        all.push_back(d);
    }

    std::vector<Det> kept = nms(all, IOU_THR);

    jfloatArray result = env->NewFloatArray(kept.size() * 6);
    std::vector<float> buf;
    for (auto& d : kept) {
        buf.insert(buf.end(), {d.x1, d.y1, d.x2, d.y2, d.conf, (float) d.cls});
    }
    env->SetFloatArrayRegion(result, 0, buf.size(), buf.data());
    return result;
}
