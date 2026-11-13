# Suppress TensorFlow verbose — harus di paling atas
import os
import sys
import logging
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

"""
evaluate.py — Versi 4 (Protokol Rössler et al. 2019)
======================================================
Evaluasi 4 model CNN pada test set FF++ C23 Deepfakes.

PERBEDAAN dari versi sebelumnya:
  - Input size per model (bukan semua 224×224)
    Xception    : 299×299 (standar paper FF++)
    ResNet50    : 224×224
    MobileNetV2 : 224×224
    MesoNet4    : 256×256 (standar paper MesoNet)
  - Generator dibuat ulang per model sesuai input size
  - Warm-up menggunakan dummy dengan ukuran yang benar
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

# ─────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────
DATA_DIR    = "data"
MODEL_DIR   = "models"
RESULTS_DIR = "results/metrics"

BATCH_SIZE  = 32

# Input size per model — HARUS sesuai dengan yang dipakai saat training
MODEL_INPUT_SIZE = {
    "Xception":    (299, 299),
    "ResNet50":    (224, 224),
    "MobileNetV2": (224, 224),
    "MesoNet4":    (256, 256),
}

MODEL_FILES = {
    "ResNet50":    "resnet50_best.keras",
    "Xception":    "xception_best.keras",
    "MobileNetV2": "mobilenetv2_best.keras",
    "MesoNet4":    "mesonet4_best.keras",
}

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────
# VALIDASI PATH
# ─────────────────────────────────────────────────────

def validate_paths():
    print("\n[VALIDASI PATH]")
    all_ok = True

    for label in ["fake", "real"]:
        folder = os.path.join(DATA_DIR, "test", label)
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            print(f"  OK  data/test/{label}: {count:,} file")
            if count == 0:
                print(f"  WARN: folder kosong!")
                all_ok = False
        else:
            print(f"  MISSING  data/test/{label}/")
            all_ok = False

    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  OK  models/{fname} ({size_mb:.1f} MB)")
        else:
            print(f"  MISSING  models/{fname}")
            all_ok = False

    if not all_ok:
        print("\n[ERROR] Ada path yang bermasalah.")
        sys.exit(1)

    print("  Semua path valid.")


# ─────────────────────────────────────────────────────
# DATA GENERATOR — PER MODEL
# ─────────────────────────────────────────────────────

def build_test_generator(model_name):
    """
    Buat test generator dengan target_size sesuai model.
    Setiap model punya input size berbeda sesuai protokol paper.
    """
    img_size = MODEL_INPUT_SIZE[model_name]
    gen      = ImageDataGenerator(rescale=1.0 / 255)

    test_data = gen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    print(f"  Input size    : {img_size}")
    print(f"  Test samples  : {test_data.samples:,}")
    print(f"  Class indices : {test_data.class_indices}")
    return test_data


# ─────────────────────────────────────────────────────
# EVALUASI SATU MODEL
# ─────────────────────────────────────────────────────

def evaluate_model(model_name, model_path):
    """
    Load model → buat generator dengan input size yang benar
    → prediksi → hitung semua metrik.
    """
    print(f"\n{'─'*50}")
    print(f"  Evaluasi: {model_name}")

    # Load model
    model = load_model(model_path)
    print(f"  Parameters: {model.count_params():,}")
    print(f"  Input shape: {model.input_shape}")

    # Buat generator dengan input size yang sesuai model ini
    test_data = build_test_generator(model_name)

    # Warm-up dengan dummy seukuran input model
    img_size = MODEL_INPUT_SIZE[model_name]
    dummy    = np.zeros((1, *img_size, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)

    # Prediksi
    test_data.reset()
    y_true = test_data.classes.copy()

    t_start     = time.time()
    y_pred_prob = model.predict(test_data, verbose=0).flatten()
    t_elapsed   = time.time() - t_start
    inference_ms = round((t_elapsed / len(y_true)) * 1000, 3)

    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Orientasi label:
    # ImageDataGenerator alfabet → fake=0, real=1
    # Flip agar fake=1 (positive class)
    class_indices = test_data.class_indices
    if class_indices.get("fake") == 0:
        y_true_eval = 1 - y_true
        y_pred_eval = 1 - y_pred
        y_prob_eval = 1 - y_pred_prob
    else:
        y_true_eval = y_true
        y_pred_eval = y_pred
        y_prob_eval = y_pred_prob

    # Hitung metrik
    acc       = float(accuracy_score(y_true_eval, y_pred_eval))
    precision = float(precision_score(y_true_eval, y_pred_eval, zero_division=0))
    recall    = float(recall_score(y_true_eval, y_pred_eval, zero_division=0))
    f1        = float(f1_score(y_true_eval, y_pred_eval, zero_division=0))
    auc       = float(roc_auc_score(y_true_eval, y_prob_eval))
    cm        = confusion_matrix(y_true_eval, y_pred_eval)
    fpr, tpr, _ = roc_curve(y_true_eval, y_prob_eval)

    total_params = int(model.count_params())

    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  AUC-ROC    : {auc:.4f}")
    print(f"  Inference  : {inference_ms:.3f} ms/frame")
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"  CM         : TP={tp} TN={tn} FP={fp} FN={fn}")

    tf.keras.backend.clear_session()

    return {
        "model":          model_name,
        "accuracy":       round(acc, 4),
        "precision":      round(precision, 4),
        "recall":         round(recall, 4),
        "f1_score":       round(f1, 4),
        "auc_roc":        round(auc, 4),
        "inference_ms":   inference_ms,
        "total_params":   total_params,
        "n_samples":      int(len(y_true)),
        "confusion_matrix": cm.tolist(),
        "roc_fpr":        fpr.tolist(),
        "roc_tpr":        tpr.tolist(),
    }


# ─────────────────────────────────────────────────────
# VISUALISASI
# ─────────────────────────────────────────────────────

def plot_comparison_metrics(results):
    df      = pd.DataFrame(results)
    metrics = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle(
        "Classification Performance — 4 CNN Models\n"
        "FaceForensics++ C23 Deepfakes · Protokol Rössler et al. 2019",
        fontsize=12, fontweight="bold", y=1.02
    )

    for ax, metric, label in zip(axes, metrics, labels):
        vals = df[metric].values
        bars = ax.bar(df["model"], vals, color=COLORS[:len(vals)],
                      edgecolor="white", linewidth=0.8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score")
        ax.set_xticklabels(df["model"], rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.axhline(y=0.5, color="gray", linestyle=":", lw=1, alpha=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "comparison_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: comparison_metrics.png")


def plot_confusion_matrices(results):
    n    = len(results)
    cols = 2
    rows = (n + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))
    fig.suptitle(
        "Confusion Matrices — 4 CNN Models\n"
        "Positive class = Fake | Protokol Rössler et al. 2019",
        fontsize=12, fontweight="bold"
    )
    axes = axes.flatten() if n > 1 else [axes]

    for i, res in enumerate(results):
        cm = np.array(res["confusion_matrix"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "Fake"],
            yticklabels=["Real", "Fake"],
            ax=axes[i], cbar=False, linewidths=0.5,
            annot_kws={"size": 14, "weight": "bold"},
        )
        axes[i].set_title(
            f"{res['model']}\n"
            f"Acc={res['accuracy']:.3f}  "
            f"F1={res['f1_score']:.3f}  "
            f"AUC={res['auc_roc']:.3f}",
            fontsize=10
        )
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            axes[i].text(
                0.5, -0.16,
                f"TP={tp}  TN={tn}  FP={fp}  FN={fn}",
                transform=axes[i].transAxes,
                ha="center", fontsize=9, color="#555"
            )

    # Sembunyikan subplot kosong jika jumlah model ganjil
    for j in range(len(results), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(pad=2.5)
    path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: confusion_matrices.png")


def plot_roc_curves(results):
    fig, ax = plt.subplots(figsize=(8, 7))

    for res, color in zip(results, COLORS):
        ax.plot(
            res["roc_fpr"], res["roc_tpr"],
            color=color, lw=2.5,
            label=f"{res['model']} (AUC={res['auc_roc']:.3f})"
        )

    ax.plot([0,1], [0,1], "k--", lw=1.5,
            label="Random (AUC=0.500)", alpha=0.6)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "ROC Curve — Comparative Analysis\n"
        "FaceForensics++ C23 Deepfakes · Protokol Rössler et al. 2019",
        fontsize=12, fontweight="bold"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)

    path = os.path.join(RESULTS_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: roc_curves.png")


def plot_efficiency(results):
    names     = [r["model"] for r in results]
    inf_times = [r["inference_ms"] for r in results]
    params_m  = [r["total_params"] / 1e6 for r in results]
    colors    = COLORS[:len(results)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Model Efficiency Comparison\n"
                 "Protokol Rössler et al. 2019",
                 fontsize=12, fontweight="bold")

    for ax, vals, title, ylabel, suffix in zip(
        axes,
        [inf_times, params_m],
        ["Inference Time (ms/frame)", "Total Parameters (Millions)"],
        ["Milliseconds", "Parameters (M)"],
        ["ms", "M"]
    ):
        bars = ax.bar(names, vals, color=colors, edgecolor="white")
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                val + max(vals) * 0.02,
                f"{val:.2f}{suffix}",
                ha="center", va="bottom", fontsize=9
            )

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "efficiency_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: efficiency_comparison.png")


def save_csv(results):
    rows = [{
        "Model":           r["model"],
        "Accuracy":        r["accuracy"],
        "Precision":       r["precision"],
        "Recall":          r["recall"],
        "F1-Score":        r["f1_score"],
        "AUC-ROC":         r["auc_roc"],
        "Inference_ms":    r["inference_ms"],
        "Total_Params":    r["total_params"],
        "Input_Size":      "x".join(map(str, MODEL_INPUT_SIZE[r["model"]])),
        "Test_Samples":    r["n_samples"],
    } for r in results]

    df   = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: comparison_table.csv")
    return df


def print_final_table(df):
    sep = "=" * 90
    print(f"\n{sep}")
    print("TABEL PERBANDINGAN LENGKAP")
    print(sep)
    cols = ["Model", "Accuracy", "Precision", "Recall",
            "F1-Score", "AUC-ROC", "Inference_ms",
            "Input_Size", "Total_Params"]
    print(df[cols].to_string(index=False))
    print(sep)

    print("\nRANKING MODEL (berdasarkan AUC-ROC):")
    ranked = df.sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
    for i, row in ranked.iterrows():
        print(
            f"  #{i+1} {row['Model']:<14} "
            f"Acc={row['Accuracy']:.4f}  "
            f"F1={row['F1-Score']:.4f}  "
            f"AUC={row['AUC-ROC']:.4f}  "
            f"Inf={row['Inference_ms']:.2f}ms  "
            f"Params={int(row['Total_Params']):,}"
        )


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FASE 4 — EVALUASI & PERBANDINGAN 4 MODEL")
    print("Dataset  : FF++ C23 Deepfakes · Split Resmi FF++")
    print("Protokol : Rössler et al. ICCV 2019")
    print("=" * 60)
    print("\nInput size per model:")
    for name, size in MODEL_INPUT_SIZE.items():
        print(f"  {name:<14}: {size[0]}×{size[1]}")

    # 0. Validasi
    validate_paths()

    # 1. Evaluasi setiap model dengan generator yang sesuai
    print("\n[1/4] Evaluasi model...")
    results = []

    for model_name, model_file in MODEL_FILES.items():
        model_path = os.path.join(MODEL_DIR, model_file)
        try:
            result = evaluate_model(model_name, model_path)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] {model_name}: {e}")
            continue

    if not results:
        print("\n[ERROR] Tidak ada model yang berhasil dievaluasi.")
        return

    # 2. Simpan JSON mentah
    raw = [{k: v for k, v in r.items()
            if k not in ("roc_fpr", "roc_tpr")}
           for r in results]
    with open(os.path.join(RESULTS_DIR, "evaluation_report.json"), "w") as f:
        json.dump(raw, f, indent=2)

    # 3. Visualisasi
    print("\n[2/4] Membuat visualisasi...")
    plot_comparison_metrics(results)
    plot_confusion_matrices(results)
    plot_roc_curves(results)
    plot_efficiency(results)

    # 4. Tabel & ranking
    print("\n[3/4] Export tabel perbandingan...")
    df = save_csv(results)
    print_final_table(df)

    # 5. Status file
    print(f"\n[4/4] Status file output:")
    for fname in ["comparison_table.csv", "comparison_metrics.png",
                  "confusion_matrices.png", "roc_curves.png",
                  "efficiency_comparison.png", "evaluation_report.json"]:
        path   = os.path.join(RESULTS_DIR, fname)
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{status}] {fname}")

    print(f"\n{'='*60}")
    print("FASE 4 SELESAI")
    print(f"{'='*60}")
    print(f"Output: {os.path.abspath(RESULTS_DIR)}/")
    print("Lanjut ke: python src/gradcam.py")


if __name__ == "__main__":
    main()