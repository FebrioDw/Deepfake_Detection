"""
detector.py — Versi 2 (Sesuai Protokol Rössler et al. 2019)
=============================================================
Update dari versi sebelumnya:
  - Input size per model (Xception=299, MesoNet4=256, lainnya=224)
  - Grad-CAM kompatibel Keras 3.x (Score-CAM fallback)
  - MTCNN threshold disesuaikan
  - Confidence orientation fix (fake=0 dalam generator)
"""

import os
import cv2
import time
import base64
import logging
import warnings
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────────────
# KONFIGURASI PER MODEL
# ─────────────────────────────────────────────────────

MODEL_CONFIG = {
    "resnet50": {
        "file":       "resnet50_best.keras",
        "input_size": (224, 224),
        "name":       "ResNet50",
        "params":     "23.6M",
    },
    "xception": {
        "file":       "xception_best.keras",
        "input_size": (299, 299),
        "name":       "Xception",
        "params":     "20.9M",
    },
    "mobilenetv2": {
        "file":       "mobilenetv2_best.keras",
        "input_size": (224, 224),
        "name":       "MobileNetV2",
        "params":     "2.3M",
    },
    "mesonet4": {
        "file":       "mesonet4_best.keras",
        "input_size": (256, 256),
        "name":       "MesoNet4",
        "params":     "28K",
    },
}

# Layer Grad-CAM per model
GRADCAM_LAYERS = {
    "resnet50":    ["conv5_block3", "conv5"],
    "xception":    ["block14_sepconv2", "sepconv2"],
    "mobilenetv2": ["Conv_1", "block_16"],
    "mesonet4":    ["conv2d"],
}

FRAMES_PER_VIDEO = 10
FACE_MARGIN      = 0.3
FAKE_THRESHOLD   = 0.5


# ─────────────────────────────────────────────────────
# GRAD-CAM — KERAS 3.X COMPATIBLE
# ─────────────────────────────────────────────────────

def find_conv_layer(model, keywords):
    for kw in keywords:
        matches = [l for l in model.layers
                   if kw in l.name and hasattr(l, "filters")]
        if matches:
            return matches[-1]
    conv_types = (tf.keras.layers.Conv2D,
                  tf.keras.layers.SeparableConv2D,
                  tf.keras.layers.DepthwiseConv2D)
    conv_layers = [l for l in model.layers if isinstance(l, conv_types)]
    return conv_layers[-1] if conv_layers else None


def compute_gradcam(model, layer, img_array):
    """Score-CAM: stable untuk Keras 3.x, tidak butuh output_shape."""
    try:
        img_tensor = tf.cast(img_array, tf.float32)

        # Coba Grad-CAM standar dulu
        try:
            grad_model = tf.keras.Model(
                inputs=model.input,
                outputs=[layer.output, model.output]
            )
            with tf.GradientTape() as tape:
                conv_out, preds = grad_model(img_tensor, training=False)
                tape.watch(conv_out)
                score = preds[:, 0]
            grads = tape.gradient(score, conv_out)

            if grads is not None and tf.reduce_max(tf.abs(grads)) > 1e-8:
                weights  = tf.reduce_mean(grads, axis=(0, 1, 2))
                cam      = tf.reduce_sum(conv_out[0] * weights, axis=-1)
                cam      = tf.nn.relu(cam).numpy()
                eps      = 1e-8
                cam      = (cam - cam.min()) / (cam.max() - cam.min() + eps)
                return cam
        except Exception:
            pass

        # Fallback: Score-CAM
        feat_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
        feat_maps  = feat_model(img_tensor, training=False)[0].numpy()
        H, W, C    = feat_maps.shape
        inp_h, inp_w = img_array.shape[1], img_array.shape[2]

        base_score = float(model(img_tensor, training=False)[0][0])
        weights    = np.zeros(C)
        top_ch     = np.argsort(feat_maps.max(axis=(0, 1)))[-20:]

        for c in top_ch:
            fm      = feat_maps[:, :, c]
            fm_norm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
            fm_up   = cv2.resize(fm_norm, (inp_w, inp_h))
            masked  = img_tensor.numpy() * fm_up[np.newaxis, :, :, np.newaxis]
            score   = float(model(masked.astype(np.float32),
                                  training=False)[0][0])
            weights[c] = score - base_score

        cam = np.zeros((H, W))
        for c in range(C):
            if weights[c] > 0:
                cam += weights[c] * feat_maps[:, :, c]
        cam = np.maximum(cam, 0)
        eps = 1e-8
        cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)
        return cam

    except Exception as e:
        print(f"  [WARN] Grad-CAM gagal: {e}")
        return None


def apply_heatmap(heatmap, img_bgr, alpha=0.45):
    h, w   = img_bgr.shape[:2]
    hm_r   = cv2.resize(heatmap, (w, h))
    hm_u8  = np.uint8(255 * hm_r)
    hm_c   = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, hm_c, alpha, 0)
    return hm_c, overlay


def img_to_b64(img_bgr):
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ─────────────────────────────────────────────────────
# DEEPFAKE DETECTOR CLASS
# ─────────────────────────────────────────────────────

class DeepfakeDetector:

    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.models     = {}
        self.mtcnn      = None
        self._load_models()
        self._setup_mtcnn()

    def _load_models(self):
        print("Loading models...")
        for model_id, cfg in MODEL_CONFIG.items():
            path = os.path.join(self.models_dir, cfg["file"])
            if os.path.exists(path):
                try:
                    self.models[model_id] = load_model(path)
                    params = self.models[model_id].count_params()
                    print(f"  [OK] {cfg['name']}: {params:,} params")
                except Exception as e:
                    print(f"  [FAIL] {cfg['name']}: {e}")
            else:
                print(f"  [MISSING] {cfg['name']}: {path}")

    def _setup_mtcnn(self):
        try:
            from facenet_pytorch import MTCNN
            import torch
            device     = "cuda" if torch.cuda.is_available() else "cpu"
            self.mtcnn = MTCNN(
                keep_all=False,
                min_face_size=20,
                thresholds=[0.5, 0.6, 0.7],
                device=device,
                post_process=False,
            )
            print(f"  [OK] MTCNN ({device})")
        except Exception as e:
            print(f"  [WARN] MTCNN tidak tersedia: {e}")
            self.mtcnn = None

    def get_loaded_models(self):
        return list(self.models.keys())

    def _crop_face(self, frame_bgr, target_size):
        """Crop wajah dengan MTCNN, resize ke target_size."""
        if self.mtcnn is not None:
            try:
                from PIL import Image
                rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img  = Image.fromarray(rgb)
                boxes, _ = self.mtcnn.detect(pil_img)
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = boxes[0]
                    h, w = frame_bgr.shape[:2]
                    mx   = int((x2 - x1) * FACE_MARGIN)
                    my   = int((y2 - y1) * FACE_MARGIN)
                    x1   = max(0, int(x1) - mx)
                    y1   = max(0, int(y1) - my)
                    x2   = min(w, int(x2) + mx)
                    y2   = min(h, int(y2) + my)
                    crop = frame_bgr[y1:y2, x1:x2]
                    if crop.size > 0:
                        return cv2.resize(crop, target_size)
            except Exception:
                pass
        # Fallback
        return cv2.resize(frame_bgr, target_size)

    def _extract_frames(self, video_path, target_size):
        """Ekstrak FRAMES_PER_VIDEO frame dari video."""
        cap   = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], {}

        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        meta = {
            "fps":          round(fps, 1),
            "width":        width,
            "height":       height,
            "duration":     round(total / fps, 2) if fps > 0 else 0,
            "total_frames": total,
        }

        n       = min(FRAMES_PER_VIDEO, total)
        indices = np.linspace(0, total - 1, n, dtype=int)
        frames  = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            face = self._crop_face(frame, target_size)
            if face is not None:
                frames.append({"idx": int(idx), "face": face})

        cap.release()
        return frames, meta

    def detect(self, video_path: str, model_id: str) -> dict:
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' tidak tersedia. "
                             f"Tersedia: {list(self.models.keys())}")

        model      = self.models[model_id]
        cfg        = MODEL_CONFIG[model_id]
        input_size = cfg["input_size"]

        # 1. Ekstrak frame
        frames, video_meta = self._extract_frames(video_path, input_size)
        if not frames:
            raise ValueError("Tidak ada frame yang berhasil diekstrak")

        # 2. Prediksi per frame
        t_start      = time.time()
        frame_results = []

        for f in frames:
            inp  = np.expand_dims(
                f["face"].astype(np.float32) / 255.0, axis=0
            )
            prob = float(model(inp, training=False)[0][0])

            # fake=0 dalam training generator → fake conf = 1 - prob
            fake_conf = round((1 - prob) * 100, 1)
            is_fake   = (1 - prob) >= FAKE_THRESHOLD

            frame_results.append({
                "frame_idx":  f["idx"],
                "fake_prob":  fake_conf,
                "is_fake":    is_fake,
                "face_b64":   img_to_b64(f["face"]),
                "_face":      f["face"],
            })

        inference_ms = round(
            (time.time() - t_start) / len(frames) * 1000, 2
        )

        # 3. Verdict keseluruhan
        probs       = [r["fake_prob"] for r in frame_results]
        avg_conf    = round(float(np.mean(probs)), 1)
        n_fake      = sum(1 for r in frame_results if r["is_fake"])
        is_deepfake = avg_conf >= (FAKE_THRESHOLD * 100)

        # 4. Grad-CAM untuk 3 frame confidence tertinggi
        layer          = find_conv_layer(model, GRADCAM_LAYERS[model_id])
        gradcam_results = []

        top_frames = sorted(frame_results,
                            key=lambda x: x["fake_prob"],
                            reverse=True)[:3]

        for fr in top_frames:
            inp     = np.expand_dims(
                fr["_face"].astype(np.float32) / 255.0, axis=0
            )
            heatmap = compute_gradcam(model, layer, inp) if layer else None

            if heatmap is not None:
                hm_c, overlay = apply_heatmap(heatmap, fr["_face"])
                gradcam_results.append({
                    "frame_idx":    fr["frame_idx"],
                    "fake_prob":    fr["fake_prob"],
                    "original_b64": fr["face_b64"],
                    "heatmap_b64":  img_to_b64(hm_c),
                    "overlay_b64":  img_to_b64(overlay),
                })

        # 5. Bersihkan field internal
        for r in frame_results:
            del r["_face"]

        return {
            "verdict": {
                "is_deepfake":     is_deepfake,
                "confidence":      avg_conf,
                "label":           "Deepfake Detected" if is_deepfake
                                   else "Real Video",
                "frames_analyzed": len(frame_results),
                "frames_flagged":  n_fake,
                "inference_ms":    inference_ms,
            },
            "video_meta":    video_meta,
            "frame_results": frame_results,
            "gradcam":       gradcam_results,
        }