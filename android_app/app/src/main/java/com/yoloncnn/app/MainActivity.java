package com.yoloncnn.app;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.hardware.Camera;
import android.os.Bundle;
import android.app.Activity;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.TextView;

/**
 * 单 Activity 实时检测：
 * Camera 预览帧(NV21) -> Bitmap -> JNI(NCNN 推理+手写后处理) -> Overlay 画框。
 */
public class MainActivity extends Activity implements SurfaceHolder.Callback {

    private static final String[] COCO_NAMES = {
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
        "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
        "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
        "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
        "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
        "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
        "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
        "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
    };

    private Camera camera;
    private YoloNcnn yolo = new YoloNcnn();
    private OverlayView overlay;
    private TextView fpsView;
    private volatile boolean busy = false;
    private long lastTime = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        if (checkSelfPermission(Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, 0);
            return;
        }
        init();
    }

    @Override
    public void onRequestPermissionsResult(int r, String[] p, int[] g) {
        if (g.length > 0 && g[0] == PackageManager.PERMISSION_GRANTED) init();
        else finish();
    }

    private void init() {
        boolean ok = yolo.loadModel(getAssets(), "model.ncnn.param", "model.ncnn.bin");
        if (!ok) { finish(); return; }

        FrameLayout root = new FrameLayout(this);
        SurfaceView sv = new SurfaceView(this);
        sv.getHolder().addCallback(this);
        overlay = new OverlayView();
        fpsView = new TextView(this);
        fpsView.setTextColor(Color.GREEN);
        fpsView.setTextSize(14);
        fpsView.setPadding(20, 40, 0, 0);
        root.addView(sv);
        root.addView(overlay);
        root.addView(fpsView);
        setContentView(root);
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) { startCamera(holder); }
    @Override
    public void surfaceChanged(SurfaceHolder h, int f, int w, int ht) {}
    @Override
    public void surfaceDestroyed(SurfaceHolder h) { stopCamera(); }

    private void startCamera(SurfaceHolder holder) {
        try {
            camera = Camera.open(0);
            Camera.Parameters params = camera.getParameters();
            params.setPreviewSize(640, 480);
            camera.setParameters(params);
            camera.setDisplayOrientation(90);
            camera.setPreviewDisplay(holder);
            camera.setPreviewCallback(previewCallback);
            camera.startPreview();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void stopCamera() {
        if (camera != null) {
            camera.setPreviewCallback(null);
            camera.stopPreview();
            camera.release();
            camera = null;
        }
    }

    private final Camera.PreviewCallback previewCallback = (data, cam) -> {
        if (busy) return;
        busy = true;
        new Thread(() -> {
            try {
                Camera.Size s = cam.getParameters().getPreviewSize();
                int[] rgba = nv21ToRgba(data, s.width, s.height);
                Bitmap bmp = Bitmap.createBitmap(rgba, s.width, s.height,
                                                 Bitmap.Config.ARGB_8888);
                long t0 = System.currentTimeMillis();
                float[] dets = yolo.detect(bmp);
                long cost = System.currentTimeMillis() - t0;
                runOnUiThread(() -> {
                    overlay.setResults(dets, bmp.getWidth(), bmp.getHeight());
                    fpsView.setText(cost + " ms/帧");
                });
            } finally {
                busy = false;
            }
        }).start();
    };

    /** NV21 -> RGBA int 数组（标准转换，逐像素）。 */
    private static int[] nv21ToRgba(byte[] nv21, int w, int h) {
        int[] out = new int[w * h];
        int frameSize = w * h;
        for (int j = 0; j < h; j++) {
            int uvRow = frameSize + (j >> 1) * w;
            for (int i = 0; i < w; i++) {
                int y = nv21[j * w + i] & 0xff;
                int v = nv21[uvRow + (i & ~1) + 1] & 0xff;
                int u = nv21[uvRow + (i & ~1)] & 0xff;
                int r = y + (int) (1.402f * (v - 128));
                int g = y - (int) (0.344f * (u - 128) + 0.714f * (v - 128));
                int b = y + (int) (1.772f * (u - 128));
                out[j * w + i] = 0xff000000
                        | (Math.min(255, Math.max(0, r)) << 16)
                        | (Math.min(255, Math.max(0, g)) << 8)
                        | Math.min(255, Math.max(0, b));
            }
        }
        return out;
    }

    /** 检测框覆盖层。 */
    class OverlayView extends View {
        private float[] dets = new float[0];
        private int srcW = 1, srcH = 1;
        private final Paint boxPaint = new Paint();
        private final Paint textPaint = new Paint();

        OverlayView() {
            super(MainActivity.this);
            boxPaint.setColor(Color.GREEN);
            boxPaint.setStyle(Paint.Style.STROKE);
            boxPaint.setStrokeWidth(4);
            textPaint.setColor(Color.GREEN);
            textPaint.setTextSize(32);
        }

        void setResults(float[] d, int w, int h) {
            dets = d; srcW = w; srcH = h;
            postInvalidate();
        }

        @Override
        protected void onDraw(Canvas canvas) {
            float scale = Math.min((float) getWidth() / srcH, (float) getHeight() / srcW);
            for (int i = 0; i + 5 < dets.length; i += 6) {
                // 预览旋转了 90°（竖屏），坐标系换成 (h, w) 再镜像到屏幕
                float x1 = dets[i], y1 = dets[i + 1], x2 = dets[i + 2], y2 = dets[i + 3];
                float conf = dets[i + 4];
                int cls = (int) dets[i + 5];
                float sx1 = y1 * scale, sy1 = x1 * scale;
                float sx2 = y2 * scale, sy2 = x2 * scale;
                canvas.drawRect(new RectF(sx1, sy1, sx2, sy2), boxPaint);
                String name = cls < COCO_NAMES.length ? COCO_NAMES[cls] : String.valueOf(cls);
                canvas.drawText(String.format("%s %.2f", name, conf), sx1, sy1 - 6, textPaint);
            }
        }
    }
}
