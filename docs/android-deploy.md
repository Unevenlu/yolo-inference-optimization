# Android 部署：YOLO-NCNN 实时检测 App

**日期**: 2026-10-15

## 产物

`android_app/app/build/outputs/apk/debug/app-debug.apk`（约 21MB，arm64-v8a）

安装方式：

- 把 APK 拷到手机，点击安装（需允许"安装未知来源应用"），或
- 手机开 USB 调试连电脑：`adb install app-debug.apk`（adb 在 `D:\tools\android-sdk\platform-tools\`）

首次打开需授予相机权限。打开后即为摄像头实时检测画面，左上角显示每帧耗时。

## 架构

```
相机预览帧(NV21)
  └─ MainActivity.java: NV21 → RGBA Bitmap（Java 层）
      └─ JNI: yolo_ncnn_jni.cpp（对应实验 008 的手写管线）
          ├─ letterbox 前处理（等比缩放 + 114 填充）
          ├─ NCNN 推理（4 线程，arm64）
          ├─ 解码 (84×8400) + 反 letterbox
          └─ 手写 NMS（按类别 IoU 0.45）
      └─ OverlayView 画框 + 类别/置信度
```

- 模型：`yolov8n` NCNN 导出（实验 008 产物，FP32，13MB）
- 引擎：NCNN 20260526 预编译静态库（含 Vulkan，默认 CPU 4 线程）
- 相机：旧版 `android.hardware.Camera` API（ deprecated 但简单可靠，demo 够用）

## 重新构建（无 Android Studio）

```bash
# 环境（已安装在 D:\tools）
export JAVA_HOME="D:\conda\miniconda3\envs\yolo\Library"
export ANDROID_SDK_ROOT="D:\tools\android-sdk"
export PATH="$JAVA_HOME/bin:/d/tools/gradle-8.9/bin:$PATH"

cd android_app
gradle assembleDebug --no-daemon
# 产物: app/build/outputs/apk/debug/app-debug.apk
```

## 换模型

1. `python -c "from ultralytics import YOLO; YOLO('你的模型.pt').export(format='ncnn', imgsz=640)"`
2. 把 `*_ncnn_model/` 里的 `model.ncnn.param` / `model.ncnn.bin` 拷到 `app/src/main/assets/`
3. 重新构建

## 已知简化（可改进点）

- 预览分辨率固定 640×480，竖屏坐标映射用了简化的旋转换算
- NV21→RGBA 转换在 Java 层逐像素做，有优化空间（可下沉到 C++/NEON）
- 未启用 Vulkan GPU：在 `loadModel` 里加 `g_net.opt.use_vulkan_compute = true` 即可对比 Adreno GPU 速度
- 线程数固定 4：骁龙 8 Elite Gen 5 可试 6/8 线程

## 本机工具链位置（不进 git）

| 工具 | 路径 |
|------|------|
| JDK 17 | `D:\conda\miniconda3\envs\yolo\Library` |
| Android SDK/NDK/CMake | `D:\tools\android-sdk` |
| Gradle 8.9 | `D:\tools\gradle-8.9` |
| NCNN 安卓库 | `D:\tools\ncnn-20260526-android-vulkan` |
