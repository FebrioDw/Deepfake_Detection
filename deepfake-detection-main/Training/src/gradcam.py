import os, sys, logging, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import cv2
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

DATA_DIR    = "data"
MODEL_DIR   = "models"
RESULTS_DIR = "results/gradcam"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_INPUT_SIZE = {
    "Xception":    (299, 299),
    "ResNet50":    (224, 224),
    "MobileNetV2": (224, 224),
    "MesoNet4":    (256, 256),
}

# Layer Grad-CAM — pakai index layer terakhir yang conv
GRADCAM_LAYER_KEYWORDS = {
    "Xception":    ["sepconv2", "block14"],
    "ResNet50":    ["conv5_block3", "conv5"],
    "MobileNetV2": ["Conv_1", "block_16"],
    "MesoNet4":    ["conv2d"],
}

MODEL_FILES = {
    "ResNet50":    "resnet50_best.keras",
    "Xception":    "xception_best.keras",
    "MobileNetV2": "mobilenetv2_best.keras",
    "MesoNet4":    "mesonet4_best.keras",
}

N_FRAMES = 4


def find_target_layer(model, keywords):
    """
    Cari layer conv yang cocok dengan keywords.
    Kembalikan layer terakhir yang match.
    """
    # Cari berdasarkan keyword (dari spesifik ke umum)
    for kw in keywords:
        matches = [l for l in model.layers
                   if kw in l.name and hasattr(l, 'filters')]
        if matches:
            layer = matches[-1]
            print(f"    Layer target: {layer.name}")
            return layer

    # Fallback: semua Conv2D/SeparableConv2D terakhir
    conv_types = (tf.keras.layers.Conv2D,
                  tf.keras.layers.SeparableConv2D,
                  tf.keras.layers.DepthwiseConv2D)
    conv_layers = [l for l in model.layers if isinstance(l, conv_types)]
    if conv_layers:
        layer = conv_layers[-1]
        print(f"    Fallback layer: {layer.name}")
        return layer
    return None


def compute_gradcam(model, layer, img_array):
    """
    Grad-CAM menggunakan tf.GradientTape dengan cara yang
    kompatibel dengan Keras 3.x — tanpa akses output_shape.
    """
    try:
        # Build sub-model dengan functional API
        # Gunakan layer.output langsung (Keras 3.x compatible)
        grad_model = tf.keras.Model(
            inputs=model.input,
            outputs=[layer.output, model.output]
        )

        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_tensor, training=False)
            tape.watch(conv_out)
            # Untuk binary sigmoid: gunakan output langsung
            score = preds[:, 0]

        grads = tape.gradient(score, conv_out)

        if grads is None or tf.reduce_max(tf.abs(grads)) < 1e-8:
            return compute_gradcam_v2(model, layer, img_array)

        # Pool gradient → weight per channel
        weights  = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam      = tf.reduce_sum(conv_out[0] * weights, axis=-1)
        cam      = tf.nn.relu(cam).numpy()

        eps = 1e-8
        cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)
        return cam

    except Exception as e:
        print(f"    [WARN] Grad-CAM v1 gagal: {type(e).__name__}: {str(e)[:80]}")
        return compute_gradcam_v2(model, layer, img_array)


def compute_gradcam_v2(model, layer, img_array):
    """
    Alternatif: Score-CAM sederhana — tidak bergantung pada gradient.
    Lebih stabil untuk model Keras 3.x.
    """
    try:
        img_tensor = tf.cast(img_array, tf.float32)

        # Ambil feature maps dari layer target
        feat_model  = tf.keras.Model(inputs=model.input,
                                      outputs=layer.output)
        feat_maps   = feat_model(img_tensor, training=False)[0]  # (H, W, C)
        feat_maps   = feat_maps.numpy()

        # Untuk setiap channel, ukur kontribusinya terhadap prediksi
        H, W, C     = feat_maps.shape
        inp_h, inp_w = img_array.shape[1], img_array.shape[2]

        weights     = np.zeros(C)
        base_score  = float(model(img_tensor, training=False)[0][0])

        # Ambil top-20 channel berdasarkan max activation (lebih cepat)
        channel_max = feat_maps.max(axis=(0, 1))
        top_ch      = np.argsort(channel_max)[-20:]

        for c in top_ch:
            # Mask input dengan feature map channel ini
            fm = feat_maps[:, :, c]
            fm_norm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
            fm_up   = cv2.resize(fm_norm, (inp_w, inp_h))
            masked  = img_tensor.numpy() * fm_up[np.newaxis, :, :, np.newaxis]
            score   = float(model(masked.astype(np.float32), training=False)[0][0])
            weights[c] = score - base_score

        # Weighted sum feature maps
        cam = np.zeros((H, W))
        for c in range(C):
            if weights[c] > 0:
                cam += weights[c] * feat_maps[:, :, c]

        cam = np.maximum(cam, 0)
        eps = 1e-8
        cam = (cam - cam.min()) / (cam.max() - cam.min() + eps)
        return cam

    except Exception as e:
        print(f"    [WARN] Score-CAM juga gagal: {type(e).__name__}: {str(e)[:80]}")
        return None


def make_overlay(heatmap, img_bgr, target_size, alpha=0.45):
    img_r   = cv2.resize(img_bgr, target_size)
    hm_r    = cv2.resize(heatmap, target_size)
    hm_u8   = np.uint8(255 * hm_r)
    hm_c    = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_r, 1 - alpha, hm_c, alpha, 0)
    return hm_c, overlay


def select_frames(n=N_FRAMES):
    fake_dir = os.path.join(DATA_DIR, "test", "fake")
    files    = sorted([f for f in os.listdir(fake_dir)
                       if f.lower().endswith(".jpg")])
    indices  = np.linspace(0, len(files) - 1, n, dtype=int)
    frames   = []
    for idx in indices:
        path    = os.path.join(fake_dir, files[idx])
        img_bgr = cv2.imread(path)
        if img_bgr is not None:
            frames.append({"path": path, "filename": files[idx],
                           "img_bgr": img_bgr})
    print(f"  {len(frames)} frame dipilih:")
    for i, f in enumerate(frames):
        print(f"    #{i+1}: {f['filename']}")
    return frames


def plot_per_model(model_name, model, layer, frames):
    n = len(frames)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.5 * n))
    if n == 1: axes = [axes]

    layer_name = layer.name if layer else "N/A"
    fig.suptitle(
        f"Grad-CAM — {model_name}\n"
        f"Target layer: {layer_name} | FF++ C23 | Protokol Rössler et al. 2019",
        fontsize=12, fontweight="bold"
    )

    col_titles = ["Original (Fake)", "Grad-CAM Heatmap", "Overlay"]
    for j, t in enumerate(col_titles):
        axes[0][j].set_title(t, fontsize=11, fontweight="bold")

    img_size = MODEL_INPUT_SIZE[model_name]

    for i, frame in enumerate(frames):
        img_bgr   = cv2.resize(frame["img_bgr"], img_size)
        img_norm  = img_bgr.astype(np.float32) / 255.0
        img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inp       = np.expand_dims(img_norm, axis=0)

        prob  = float(model(inp, training=False)[0][0])
        conf  = round(1 - prob, 3)

        heatmap = compute_gradcam(model, layer, inp) if layer else None

        # Original
        axes[i][0].imshow(img_rgb)
        axes[i][0].set_ylabel(f"Frame {i+1}\nFake conf: {conf}",
                              fontsize=9, fontweight="bold")
        axes[i][0].set_xticks([]); axes[i][0].set_yticks([])

        if heatmap is not None:
            hm_c, overlay = make_overlay(heatmap, frame["img_bgr"], img_size)

            # Heatmap
            im = axes[i][1].imshow(
                cv2.resize(heatmap, img_size), cmap="jet", vmin=0, vmax=1)
            plt.colorbar(im, ax=axes[i][1], fraction=0.046, pad=0.04)
            axes[i][1].set_xticks([]); axes[i][1].set_yticks([])

            # Overlay
            axes[i][2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[i][2].set_xticks([]); axes[i][2].set_yticks([])
        else:
            for j in [1, 2]:
                axes[i][j].text(0.5, 0.5, "N/A", ha="center", va="center",
                                transform=axes[i][j].transAxes, fontsize=12)
                axes[i][j].set_xticks([]); axes[i][j].set_yticks([])

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{model_name.lower()}_gradcam.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {model_name.lower()}_gradcam.png")


def plot_comparison(all_results, frames):
    """
    Layout: N_FRAMES baris × (1 original + N_models overlay) kolom
    Semua model pada frame yang sama — figure utama paper.
    """
    n_frames = len(frames)
    n_models = len(all_results)
    n_cols   = 1 + n_models

    fig, axes = plt.subplots(n_frames, n_cols,
                              figsize=(3.8 * n_cols, 4.2 * n_frames))
    if n_frames == 1: axes = [axes]

    fig.suptitle(
        "Grad-CAM Comparison — 4 CNN Architectures\n"
        "FaceForensics++ C23 Deepfakes | Same Input Frame | Protokol Rössler et al. 2019\n"
        "Warmer colors = higher activation (model attention regions)",
        fontsize=12, fontweight="bold"
    )

    # Header kolom
    axes[0][0].set_title("Original\n(Fake Frame)", fontsize=10, fontweight="bold")
    for j, (name, _, _) in enumerate(all_results):
        axes[0][j+1].set_title(f"{name}\nOverlay", fontsize=10, fontweight="bold")

    DISPLAY_SIZE = (224, 224)

    for i, frame in enumerate(frames):
        # Kolom 0: original
        disp = cv2.resize(frame["img_bgr"], DISPLAY_SIZE)
        axes[i][0].imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        axes[i][0].set_ylabel(f"Frame {i+1}", fontsize=9, fontweight="bold")
        axes[i][0].set_xticks([]); axes[i][0].set_yticks([])

        for j, (model_name, model, layer) in enumerate(all_results):
            img_size = MODEL_INPUT_SIZE[model_name]
            img_bgr  = cv2.resize(frame["img_bgr"], img_size)
            img_norm = img_bgr.astype(np.float32) / 255.0
            inp      = np.expand_dims(img_norm, axis=0)
            prob     = float(model(inp, training=False)[0][0])
            conf     = round(1 - prob, 3)
            heatmap  = compute_gradcam(model, layer, inp) if layer else None

            if heatmap is not None:
                _, overlay = make_overlay(heatmap, frame["img_bgr"], DISPLAY_SIZE)
                axes[i][j+1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                axes[i][j+1].set_xlabel(f"fake conf={conf}", fontsize=8)
            else:
                axes[i][j+1].text(0.5, 0.5, "N/A", ha="center", va="center",
                                   transform=axes[i][j+1].transAxes)
            axes[i][j+1].set_xticks([]); axes[i][j+1].set_yticks([])

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "comparison_gradcam.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: comparison_gradcam.png")


def plot_paper_summary(all_results, frame):
    """
    Figure untuk paper: 3 baris × 4 kolom
    Baris 1: Original
    Baris 2: Heatmap (jet)
    Baris 3: Overlay
    """
    n_models = len(all_results)
    fig, axes = plt.subplots(3, n_models, figsize=(4.2 * n_models, 12))

    fig.suptitle(
        "Grad-CAM Explainability — 4 CNN Architectures for Deepfake Detection\n"
        "FaceForensics++ C23 Deepfakes | XAI Visual Explanation",
        fontsize=13, fontweight="bold"
    )

    row_labels = ["Original Frame", "Grad-CAM Heatmap", "Heatmap Overlay"]
    for i, lb in enumerate(row_labels):
        axes[i][0].set_ylabel(lb, fontsize=11, fontweight="bold")

    DISPLAY_SIZE = (256, 256)

    for j, (model_name, model, layer) in enumerate(all_results):
        img_size = MODEL_INPUT_SIZE[model_name]
        img_bgr  = cv2.resize(frame["img_bgr"], img_size)
        img_norm = img_bgr.astype(np.float32) / 255.0
        inp      = np.expand_dims(img_norm, axis=0)
        prob     = float(model(inp, training=False)[0][0])
        conf     = round(1 - prob, 3)
        heatmap  = compute_gradcam(model, layer, inp) if layer else None

        layer_name = layer.name.split("_")[-1] if layer else "N/A"
        axes[0][j].set_title(
            f"{model_name}\nLayer: ...{layer_name}\nconf={conf}",
            fontsize=9, fontweight="bold"
        )

        # Baris 0: original
        disp_orig = cv2.resize(frame["img_bgr"], DISPLAY_SIZE)
        axes[0][j].imshow(cv2.cvtColor(disp_orig, cv2.COLOR_BGR2RGB))
        axes[0][j].set_xticks([]); axes[0][j].set_yticks([])

        if heatmap is not None:
            hm_c, overlay = make_overlay(heatmap, frame["img_bgr"], DISPLAY_SIZE)

            # Baris 1: heatmap
            hm_up = cv2.resize(heatmap, DISPLAY_SIZE)
            im    = axes[1][j].imshow(hm_up, cmap="jet", vmin=0, vmax=1)
            plt.colorbar(im, ax=axes[1][j], fraction=0.046, pad=0.04)
            axes[1][j].set_xticks([]); axes[1][j].set_yticks([])

            # Baris 2: overlay
            axes[2][j].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[2][j].set_xticks([]); axes[2][j].set_yticks([])
        else:
            for row in [1, 2]:
                axes[row][j].text(0.5, 0.5, "N/A", ha="center", va="center",
                                   transform=axes[row][j].transAxes)
                axes[row][j].set_xticks([]); axes[row][j].set_yticks([])

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "paper_gradcam_summary.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: paper_gradcam_summary.png")


def main():
    print("=" * 60)
    print("FASE 5 — GRAD-CAM (Versi 4 — Keras 3.x Compatible)")
    print("=" * 60)

    print("\n[1/3] Pilih shared frames...")
    frames = select_frames(n=N_FRAMES)

    print("\n[2/3] Load model & Grad-CAM...")
    all_results = []

    for model_name, model_file in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, model_file)
        if not os.path.exists(path):
            print(f"\n  [SKIP] {model_name}")
            continue

        print(f"\n{'─'*50}")
        print(f"  MODEL: {model_name}")
        model = load_model(path)
        print(f"  Parameters: {model.count_params():,}")

        # List semua layer conv untuk debugging
        conv_layers = [l.name for l in model.layers
                       if hasattr(l, 'filters')]
        print(f"  Conv layers (last 5): {conv_layers[-5:]}")

        layer = find_target_layer(model,
                                   GRADCAM_LAYER_KEYWORDS[model_name])
        plot_per_model(model_name, model, layer, frames)
        all_results.append((model_name, model, layer))

    print("\n[3/3] Comparison figures...")
    if len(all_results) >= 2:
        plot_comparison(all_results, frames)
        plot_paper_summary(all_results, frames[0])

    print(f"\n{'='*60}")
    print("FASE 5 SELESAI")
    print(f"{'='*60}")
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith(".png"):
            sz = os.path.getsize(os.path.join(RESULTS_DIR, f))
            print(f"  {f:<45} ({sz/1024:.0f} KB)")


if __name__ == "__main__":
    main()