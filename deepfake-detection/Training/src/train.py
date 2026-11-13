"""
train.py — Versi 3 (Protokol Rössler et al. ICCV 2019)
========================================================
Mengikuti protokol training PERSIS paper FF++ asli:

  Phase A → Pretrain classification head saja (3 epoch, base frozen)
  Phase B → Train full network (15 epoch, semua layer unfrozen)
  Optimizer: Adam lr=0.0002
  Batch size: 32
  Pilih model dengan val_accuracy terbaik (ModelCheckpoint)
  Tidak ada ReduceLROnPlateau (paper tidak menyebutkannya)

Referensi protokol:
  "Starting from models pretrained on ImageNet, the classification
  layer is separately pretrained for 3 epochs, and then the full
  network is trained for 15 epochs and the model with best
  validation accuracy is chosen. Samples are grouped in batches
  of 32 and Adam optimizer is applied with learning rate 0.0002."
  — Replication study of Rössler et al. protocol (PMC 2021)

PERBEDAAN dari versi sebelumnya:
  - LR         : 1e-4         → 0.0002  (protokol paper)
  - Phase A    : 30 epoch     → 3 epoch (protokol paper)
  - Phase B    : fine-tune 20 → full unfreeze 15 epoch (protokol paper)
  - Input size : 224×224      → 299×299 Xception / 224×224 lainnya
  - Head       : Dense(256)+BN+Dropout → GAP+Dense(1) minimal (protokol paper)
  - Tidak ada ReduceLROnPlateau
  - EarlyStopping patience diperbesar (paper tidak menyebutkan early stop)
"""

import os
import sys
import json
import time
import logging
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50, Xception, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─────────────────────────────────────────────────────
# KONFIGURASI — PROTOKOL RÖSSLER ET AL. 2019
# ─────────────────────────────────────────────────────
DATA_DIR    = "data"
MODEL_DIR   = "models"
RESULTS_DIR = "results/metrics"

BATCH_SIZE  = 32       # persis paper
LR          = 0.0002   # Adam lr, persis paper
EPOCHS_HEAD = 3        # Phase A: head only, persis paper
EPOCHS_FULL = 15       # Phase B: full network, persis paper
RANDOM_SEED = 42

# Input size per model
MODEL_INPUT_SIZE = {
    "Xception":    (299, 299),   # paper asli FF++ pakai 299 untuk Xception
    "ResNet50":    (224, 224),
    "MobileNetV2": (224, 224),
    "MesoNet4":    (256, 256),   # MesoNet paper pakai 256×256
}

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────
# DATA GENERATOR
# ─────────────────────────────────────────────────────

def build_generators(model_name):
    """
    Protokol paper FF++:
    - rescale 1/255
    - horizontal flip saja (augmentasi minimal)
    - target_size disesuaikan per model
    """
    img_size  = MODEL_INPUT_SIZE[model_name]
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        seed=RANDOM_SEED,
        shuffle=True,
    )
    val_data = val_gen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )

    print(f"  Input size : {img_size}")
    print(f"  Train      : {train_data.samples:,} | classes: {train_data.class_indices}")
    print(f"  Val        : {val_data.samples:,}")
    return train_data, val_data


# ─────────────────────────────────────────────────────
# DEFINISI MODEL — HEAD MINIMAL SESUAI PAPER
# ─────────────────────────────────────────────────────

def build_xception():
    """
    Xception — model utama Rössler et al. 2019.
    Input 299×299. Head: GAP → Dense(1, sigmoid).
    Paper menggunakan head minimal tanpa Dropout tambahan.
    """
    img_size = MODEL_INPUT_SIZE["Xception"]
    base = Xception(weights="imagenet", include_top=False,
                    input_shape=(*img_size, 3))
    base.trainable = False
    x   = layers.GlobalAveragePooling2D()(base.output)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(base.input, out, name="Xception"), base


def build_resnet50():
    img_size = MODEL_INPUT_SIZE["ResNet50"]
    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=(*img_size, 3))
    base.trainable = False
    x   = layers.GlobalAveragePooling2D()(base.output)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(base.input, out, name="ResNet50"), base


def build_mobilenetv2():
    img_size = MODEL_INPUT_SIZE["MobileNetV2"]
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(*img_size, 3))
    base.trainable = False
    x   = layers.GlobalAveragePooling2D()(base.output)
    out = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(base.input, out, name="MobileNetV2"), base


def build_mesonet4():
    """
    MesoNet4 — dari scratch, tanpa pretrained.
    Arsitektur persis Afchar et al. WIFS 2018 dengan input 256×256.
    """
    img_size = MODEL_INPUT_SIZE["MesoNet4"]
    inp = layers.Input(shape=(*img_size, 3), name="input")

    # Block 1
    x = layers.Conv2D(8, (3, 3), padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Block 2
    x = layers.Conv2D(8, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Block 3
    x = layers.Conv2D(16, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Block 4
    x = layers.Conv2D(16, (5, 5), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((4, 4), padding="same")(x)

    # Head
    x   = layers.Flatten()(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(16)(x)
    x   = layers.LeakyReLU(negative_slope=0.1)(x)
    x   = layers.Dropout(0.5)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inp, out, name="MesoNet4"), None


MODEL_REGISTRY = {
    "Xception":    build_xception,
    "ResNet50":    build_resnet50,
    "MobileNetV2": build_mobilenetv2,
    "MesoNet4":    build_mesonet4,
}


# ─────────────────────────────────────────────────────
# COMPILE HELPER
# ─────────────────────────────────────────────────────

def compile_model(model):
    """
    Adam lr=0.0002 — persis protokol paper FF++.
    Metrics: accuracy + AUC + Precision + Recall untuk monitoring.
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


# ─────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────

def get_callbacks(model_name, phase):
    """
    Paper memilih model dengan val_accuracy terbaik.
    EarlyStopping dengan patience besar agar tidak menghentikan
    training sebelum 15 epoch (paper menyelesaikan penuh 15 epoch).
    """
    ckpt = os.path.join(MODEL_DIR, f"{model_name.lower()}_{phase}.keras")
    return [
        callbacks.ModelCheckpoint(
            filepath=ckpt,
            monitor="val_accuracy",   # paper: pilih berdasarkan val_accuracy
            save_best_only=True,
            verbose=0,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,               # besar agar tidak stop terlalu awal
            restore_best_weights=True,
            verbose=1,
        ),
    ]


# ─────────────────────────────────────────────────────
# PLOT TRAINING HISTORY
# ─────────────────────────────────────────────────────

def plot_history(hist_a, hist_b, model_name):
    def merge(key):
        a = hist_a.history.get(key, [])
        b = hist_b.history.get(key, []) if hist_b else []
        return a + b

    n_a = len(hist_a.history["loss"])
    ep  = range(1, len(merge("loss")) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Training History — {model_name}\n"
                 f"Protokol Rössler et al. 2019 | LR=0.0002 | Adam",
                 fontsize=12, fontweight="bold")

    # Loss
    axes[0].plot(ep, merge("loss"),     label="Train Loss",  color="#2196F3", lw=2)
    axes[0].plot(ep, merge("val_loss"), label="Val Loss",    color="#F44336",
                 linestyle="--", lw=2)
    if hist_b:
        axes[0].axvline(x=n_a, color="gray", linestyle=":", lw=1.5,
                        label=f"Full train start (ep {n_a})")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(ep, merge("accuracy"),     label="Train Acc", color="#4CAF50", lw=2)
    axes[1].plot(ep, merge("val_accuracy"), label="Val Acc",   color="#FF9800",
                 linestyle="--", lw=2)
    if hist_b:
        axes[1].axvline(x=n_a, color="gray", linestyle=":", lw=1.5,
                        label=f"Full train start (ep {n_a})")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy"); axes[1].set_ylim(0, 1.05)
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{model_name.lower()}_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Kurva training → {path}")


# ─────────────────────────────────────────────────────
# TRAINING SATU MODEL
# ─────────────────────────────────────────────────────

def train_one(model_name, build_fn):
    print(f"\n{'='*58}")
    print(f"  MODEL: {model_name}")
    print(f"  Protokol: Rössler et al. 2019")
    print(f"{'='*58}")

    # Load data (per model karena input size berbeda)
    print("\n  Memuat data...")
    train_data, val_data = build_generators(model_name)

    # Build model
    model, base_model = build_fn()
    print(f"  Parameters: {model.count_params():,}")

    # ── Phase A: Head Only (3 epoch) ──────────────────
    print(f"\n  [Phase A] Head-only pretraining — {EPOCHS_HEAD} epoch")
    print(f"  Base frozen | LR={LR} | Batch={BATCH_SIZE}")

    compile_model(model)

    t0 = time.time()
    hist_a = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS_HEAD,
        callbacks=get_callbacks(model_name, "phaseA"),
        verbose=1,
    )
    best_a = max(hist_a.history["val_accuracy"])
    print(f"  Phase A selesai: best val_acc={best_a:.4f} | "
          f"waktu={(time.time()-t0)/60:.1f} menit")

    # ── Phase B: Full Network (15 epoch) ──────────────
    hist_b = None
    if base_model is not None:
        print(f"\n  [Phase B] Full network training — {EPOCHS_FULL} epoch")
        print(f"  Semua layer unfrozen | LR={LR} | Batch={BATCH_SIZE}")

        # Unfreeze semua layer (protokol paper: full network)
        base_model.trainable = True
        compile_model(model)   # recompile setelah unfreeze

        t0 = time.time()
        hist_b = model.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS_FULL,
            callbacks=get_callbacks(model_name, "phaseB"),
            verbose=1,
        )
        best_b = max(hist_b.history["val_accuracy"])
        print(f"  Phase B selesai: best val_acc={best_b:.4f} | "
              f"waktu={(time.time()-t0)/60:.1f} menit")
    else:
        # MesoNet4: tidak ada base pretrained, langsung train penuh
        print(f"\n  [MesoNet4] Training dari scratch — {EPOCHS_FULL} epoch")
        # Reset dan train lebih lama untuk scratch
        model2, _ = build_fn()
        compile_model(model2)

        t0 = time.time()
        hist_b = model2.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS_FULL + EPOCHS_HEAD,  # total 18 epoch
            callbacks=get_callbacks(model_name, "full"),
            verbose=1,
        )
        model = model2
        best_b = max(hist_b.history["val_accuracy"])
        print(f"  MesoNet4 selesai: best val_acc={best_b:.4f} | "
              f"waktu={(time.time()-t0)/60:.1f} menit")
        hist_a = hist_b  # untuk plot
        hist_b = None

    # Simpan model final
    final_path = os.path.join(MODEL_DIR, f"{model_name.lower()}_best.keras")
    model.save(final_path)
    print(f"\n  Model tersimpan → {final_path}")

    # Plot
    plot_history(hist_a, hist_b, model_name)

    # Simpan history JSON
    def to_list(h):
        return {k: [float(v) for v in vals] for k, vals in h.history.items()}

    history_dict = {
        "model":    model_name,
        "protocol": "Rossler et al. ICCV 2019",
        "lr":       LR,
        "phase_a":  to_list(hist_a),
        "phase_b":  to_list(hist_b) if hist_b else {},
    }
    json_path = os.path.join(RESULTS_DIR, f"{model_name.lower()}_history.json")
    with open(json_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    return model


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("FASE 3 — TRAINING 4 MODEL CNN")
    print("Protokol: Rössler et al. ICCV 2019")
    print("Dataset : FF++ C23 Deepfakes | Split resmi 720/140/140")
    print("=" * 58)
    print(f"\nHyperparameter (persis paper):")
    print(f"  Optimizer   : Adam lr={LR}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  Phase A     : {EPOCHS_HEAD} epoch (head only, base frozen)")
    print(f"  Phase B     : {EPOCHS_FULL} epoch (full network, semua unfrozen)")
    print(f"  Model saved : best val_accuracy")

    # Info GPU
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\nGPU: {len(gpus)} unit terdeteksi")
    if not gpus:
        print("  CPU mode — estimasi: 6–10 jam untuk 4 model")
        print("  (lebih cepat dari sebelumnya karena epoch lebih sedikit)")

    # Training semua model
    start_total = time.time()

    for name, build_fn in MODEL_REGISTRY.items():
        train_one(name, build_fn)

    total_time = (time.time() - start_total) / 3600

    # Ringkasan
    print("\n" + "=" * 58)
    print("SEMUA MODEL SELESAI DITRAINING")
    print("=" * 58)
    print(f"Total waktu  : {total_time:.1f} jam")
    print(f"Model dir    : {os.path.abspath(MODEL_DIR)}/")
    print(f"Metrics dir  : {os.path.abspath(RESULTS_DIR)}/")
    print("\nFile yang dihasilkan:")
    for name in MODEL_REGISTRY:
        n = name.lower()
        print(f"  models/{n}_best.keras")
        print(f"  results/metrics/{n}_history.json")
        print(f"  results/metrics/{n}_history.png")
    print("\nLanjut ke: python src/evaluate.py")


if __name__ == "__main__":
    main()