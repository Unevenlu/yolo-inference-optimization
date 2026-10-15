package com.yoloncnn.app;

import android.graphics.Bitmap;

/** JNI 桥：模型加载与单帧检测。对应 cpp/yolo_ncnn_jni.cpp。 */
public class YoloNcnn {
    static {
        System.loadLibrary("yolo_ncnn_jni");
    }

    /** 从 assets 加载模型，返回 true 表示成功。 */
    public native boolean loadModel(android.content.res.AssetManager mgr,
                                    String paramName, String binName);

    /**
     * 对一帧 RGBA Bitmap 做检测。
     * @return float 数组，每 6 个一组: x1, y1, x2, y2, conf, clsId（原图坐标系）
     */
    public native float[] detect(Bitmap bitmap);
}
