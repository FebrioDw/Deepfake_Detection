"""
visualization_gradcam.py — Explainable AI (XAI) Pipeline
======================================================
Generates Grad-CAM / Score-CAM heatmaps for 4 CNN architectures.
Provides visual interpretability by highlighting the regions 
(pixels) the model focuses on to classify an image as a deepfake.

Features:
  - Keras 3.x compatible GradientTape implementation.
  - Score-CAM fallback mechanism for unstable gradients.
  - Generates publication-ready comparative overlays.
"""

import os
import sys
import logging
import warnings

# Suppress TensorFlow verbose logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR    = "dataset_split/test"
MODEL_DIR   = "."
RESULTS_DIR = "assets"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_INPUT_SIZE = {
    "Xception":    (299, 299),
    "ResNet50":    (224, 224),
    "MobileNetV2": (224, 224),
    "MesoNet4":    (256, 256),
}

# Target convolutional layers for heatmap extraction
GRADCAM_LAYER_KEYWORDS = {
    "Xception":    ["sepconv2", "block14"],
    "ResNet50":    ["conv5_block3", "conv5"],
    "MobileNetV2": ["Conv_1", "block_16"],
    "MesoNet4":    ["conv2d"],
}

MODEL_FILES = {
    "ResNet50":    "best_resnet50_model.h5",
    "Xception":    "best_xception_model.h5",
    "MobileNetV2": "best_mobilenet_model.h5",
    "MesoNet4":    "best_mesonet_model.h5",
}

N_FRAMES = 4

# ==========================================
# LAYER IDENTIFICATION
# ==========================================
def find_target_layer(model, keywords):
    for kw in keywords:
        matches = [l for l in model.layers if kw in l.name and hasattr(l, 'filters')]
        if matches:
            layer = matches[-1]
            print(f"    [INFO] Target layer matched: {layer.name}")
            return layer

    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D, tf.keras.layers.DepthwiseConv2D)
    conv_layers = [l for l in model.layers if isinstance(l, conv_types)]
    if conv_layers:
        layer = conv_layers[-1]
        print(f"    [WARN] Keyword missing. Using fallback layer: {layer.name}")
        return layer
    return None

# ==========================================
# HEATMAP COMPUTATION
# ==========================================
def compute_gradcam(model, layer, img_array):
    try:
        grad_model = tf.keras.Model(inputs=model.input, outputs=[layer.output, model.output])
        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_tensor, training=False)
            tape.watch(conv_out)
            score = preds[:, 0]

        grads = tape.gradient(score, conv_out)

        if grads is None or tf.reduce_max(tf.abs(grads)) < 1e-8:
            return compute_scorecam(model, layer, img_array)

        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)
        cam = tf.nn.relu(cam).numpy()

        eps = 1e-8
        cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)
        return cam

    except Exception as e:
        print(f"    [WARN] Grad-CAM failed ({type(e).__name__}). Falling back to Score-CAM.")
        return compute_scorecam(model, layer, img_array)

def compute_scorecam(model, layer, img_array):
    try:
        img_tensor = tf.cast(img_array, tf.float32)
        feat_model = tf.keras.Model(inputs=model.input, outputs=layer.output)
        feat_maps = feat_model(img_tensor, training=False)[0].numpy()

        H, W, C = feat_maps.shape
        inp_h, inp_w = img_array.shape[1], img_array.shape[2]

        weights = np.zeros(C)
        base_score = float(model(img_tensor, training=False)[0][0])

        channel_max = feat_maps.max(axis=(0, 1))
        top_ch = np.argsort(channel_max)[-20:]

        for c in top_ch:
            fm = feat_maps[:, :, c]
            fm_norm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
            fm_up = cv2.resize(fm_norm, (inp_w, inp_h))
            masked = img_tensor.numpy() * fm_up[np.newaxis, :, :, np.newaxis]
            score = float(model(masked.astype(np.float32), training=False)[0][0])
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
        print(f"    [ERROR] Score-CAM failed: {type(e).__name__}")
        return None

def make_overlay(heatmap, img_bgr, target_size, alpha=0.45):
    img_r = cv2.resize(img_bgr, target_size)
    hm_r = cv2.resize(heatmap, target_size)
    hm_u8 = np.uint8(255 * hm_r)
    hm_c = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_r, 1 - alpha, hm_c, alpha, 0)
    return hm_c, overlay

def select_frames(n=N_FRAMES):
    fake_dir = os.path.join(DATA_DIR, "fake")
    files = sorted([f for f in os.listdir(fake_dir) if f.lower().endswith(".jpg")])
    indices = np.linspace(0, len(files) - 1, n, dtype=int)
    frames = []
    
    print(f"  [INFO] Selected {n} frames for visual analysis:")
    for i, idx in enumerate(indices):
        path = os.path.join(fake_dir, files[idx])
        img_bgr = cv2.imread(path)
        if img_bgr is not None:
            frames.append({"path": path, "filename": files[idx], "img_bgr": img_bgr})
            print(f"    #{i+1}: {files[idx]}")
    return frames

# ==========================================
# VISUALIZATION MODULES
# ==========================================
def plot_comparison(all_results, frames):
    n_frames = len(frames)
    n_models = len(all_results)
    n_cols = 1 + n_models

    fig, axes = plt.subplots(n_frames, n_cols, figsize=(3.8 * n_cols, 4.2 * n_frames))
    if n_frames == 1: axes = [axes]

    fig.suptitle(
        "XAI Explainability: Grad-CAM Cross-Architecture Comparison\n"
        "Highlighting network attention regions for Deepfake manipulation detection.",
        fontsize=14, fontweight="bold", y=1.02
    )

    axes[0][0].set_title("Original Input\n(Manipulated Frame)", fontsize=11, fontweight="bold")
    for j, (name, _, _) in enumerate(all_results):
        axes[0][j+1].set_title(f"{name}\nAttention Overlay", fontsize=11, fontweight="bold")

    DISPLAY_SIZE = (224, 224)

    for i, frame in enumerate(frames):
        disp = cv2.resize(frame["img_bgr"], DISPLAY_SIZE)
        axes[i][0].imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        axes[i][0].set_ylabel(f"Frame {i+1}", fontsize=10, fontweight="bold")
        axes[i][0].set_xticks([]); axes[i][0].set_yticks([])

        for j, (model_name, model, layer) in enumerate(all_results):
            img_size = MODEL_INPUT_SIZE[model_name]
            img_bgr = cv2.resize(frame["img_bgr"], img_size)
            img_norm = img_bgr.astype(np.float32) / 255.0
            inp = np.expand_dims(img_norm, axis=0)
            
            prob = float(model(inp, training=False)[0][0])
            conf = round(1 - prob, 3)
            heatmap = compute_gradcam(model, layer, inp) if layer else None

            if heatmap is not None:
                _, overlay = make_overlay(heatmap, frame["img_bgr"], DISPLAY_SIZE)
                axes[i][j+1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                axes[i][j+1].set_xlabel(f"Fake Prob: {conf}", fontsize=9)
            else:
                axes[i][j+1].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[i][j+1].transAxes)
            axes[i][j+1].set_xticks([]); axes[i][j+1].set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "xai_gradcam_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [OK] Saved: xai_gradcam_comparison.png")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("============================================================")
    print(" INITIATING EXPLAINABLE AI (XAI) VISUALIZATION PIPELINE")
    print("============================================================")

    frames = select_frames(n=N_FRAMES)
    all_results = []

    for model_name, model_file in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, model_file)
        if not os.path.exists(path):
            print(f"\n  [SKIP] Model not found: {model_name}")
            continue

        print(f"\n{'-'*50}\n  Processing: {model_name}\n{'-'*50}")
        model = load_model(path)
        layer = find_target_layer(model, GRADCAM_LAYER_KEYWORDS[model_name])
        
        all_results.append((model_name, model, layer))

    if len(all_results) >= 2:
        print("\n[PROCESS] Generating comparative publication figures...")
        plot_comparison(all_results, frames)

    print(f"\n[SUCCESS] XAI Visualization complete. Assets saved to {os.path.abspath(RESULTS_DIR)}/")

if __name__ == "__main__":
    main()