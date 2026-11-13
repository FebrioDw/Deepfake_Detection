"""
preprocess.py — Versi 4 (Protokol Rössler et al. ICCV 2019)
=============================================================
Mengikuti protokol PERSIS paper FF++ asli:

  - 10 frame per video (uniform sampling)
  - MTCNN face detection & cropping
  - Split resmi FF++: 720 train / 140 val / 140 test
  - Target output: 7.200 / 1.400 / 1.400 frame per kelas
  - Image size: 299×299 (standar Xception, paper asli)

Referensi:
  Rössler et al. ICCV 2019 — "For every detector we have
  7200 training frames, 1400 validation frames and 1400
  testing frames for each class."

Struktur input:
    dataset/
    ├── Original_C23/
    ├── Manipulated_C23/
    └── splits/ (train.json, val.json, test.json)

Struktur output:
    data/
    ├── train/ real/ fake/
    ├── val/   real/ fake/
    └── test/  real/ fake/
"""

import os
import sys
import json
import shutil
import logging
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import cv2
import numpy as np
from tqdm import tqdm

# ─────────────────────────────────────────────────────
# KONFIGURASI — SESUAI PROTOKOL RÖSSLER ET AL. 2019
# ─────────────────────────────────────────────────────
ORIGINAL_DIR    = "dataset/Original_C23"
MANIPULATED_DIR = "dataset/Manipulated_C23"
SPLITS_DIR      = "dataset/splits"
OUTPUT_DIR      = "data"

IMG_SIZE         = (299, 299)   # standar Xception & paper asli FF++
FRAMES_PER_VIDEO = 10           # PERSIS protokol paper
FACE_MARGIN      = 0.3          # margin 30% sekitar bounding box wajah
MIN_FACE_SIZE    = 20
MTCNN_THRESHOLD  = [0.5, 0.6, 0.7]


# ─────────────────────────────────────────────────────
# SETUP FACE DETECTOR
# ─────────────────────────────────────────────────────

def setup_mtcnn():
    try:
        from facenet_pytorch import MTCNN
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mtcnn  = MTCNN(
            keep_all=False,
            min_face_size=MIN_FACE_SIZE,
            thresholds=MTCNN_THRESHOLD,
            device=device,
            post_process=False,
        )
        print(f"  MTCNN OK — device: {device}")
        return mtcnn, True
    except Exception as e:
        print(f"  MTCNN gagal ({e}), fallback ke Haar Cascade")
        haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return haar, False


# ─────────────────────────────────────────────────────
# FACE CROPPING
# ─────────────────────────────────────────────────────

def crop_face_mtcnn(frame_bgr, mtcnn):
    from PIL import Image
    rgb        = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil        = Image.fromarray(rgb)
    boxes, probs = mtcnn.detect(pil)
    if boxes is None or len(boxes) == 0:
        return None
    idx      = int(np.argmax(probs)) if probs is not None else 0
    x1,y1,x2,y2 = boxes[idx]
    h, w     = frame_bgr.shape[:2]
    mx       = int((x2 - x1) * FACE_MARGIN)
    my       = int((y2 - y1) * FACE_MARGIN)
    x1 = max(0, int(x1) - mx);  y1 = max(0, int(y1) - my)
    x2 = min(w, int(x2) + mx);  y2 = min(h, int(y2) + my)
    crop = frame_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, IMG_SIZE) if crop.size > 0 else None


def crop_face_haar(frame_bgr, haar):
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )
    if len(faces) == 0:
        return None
    x,y,w,h   = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    H,W        = frame_bgr.shape[:2]
    mx,my      = int(w*FACE_MARGIN), int(h*FACE_MARGIN)
    x1,y1      = max(0,x-mx), max(0,y-my)
    x2,y2      = min(W,x+w+mx), min(H,y+h+my)
    crop       = frame_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, IMG_SIZE) if crop.size > 0 else None


def detect_face(frame, detector, use_mtcnn):
    return crop_face_mtcnn(frame, detector) if use_mtcnn \
           else crop_face_haar(frame, detector)


# ─────────────────────────────────────────────────────
# EKSTRAKSI FRAME — PROTOKOL PAPER
# ─────────────────────────────────────────────────────

def extract_frames(video_path, out_dir, video_id, detector, use_mtcnn):
    """
    Ekstrak tepat 10 frame dengan uniform sampling (interval merata).
    Sesuai protokol Rössler et al. 2019.
    Frame yang wajahnya tidak terdeteksi di-skip (tidak ada retry).
    """
    cap   = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return 0

    n       = min(FRAMES_PER_VIDEO, total)
    indices = np.linspace(0, total - 1, n, dtype=int)
    saved   = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        face = detect_face(frame, detector, use_mtcnn)
        if face is None:
            continue
        path = os.path.join(out_dir, f"{video_id}_f{int(idx):05d}.jpg")
        cv2.imwrite(path, face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

    cap.release()
    return saved


# ─────────────────────────────────────────────────────
# SPLIT JSON
# ─────────────────────────────────────────────────────

def load_split(json_path):
    with open(json_path) as f:
        return json.load(f)


def get_video_ids(split_data):
    """
    Format JSON FF++: [["000","001"], ["000","003"], ...]
    Real : angka target unik → "000", "001", ...
    Fake : gabungan target_source → "000_001", "000_003", ...
    """
    real_ids, fake_ids = [], []
    seen = set()
    for pair in split_data:
        target, source = pair[0], pair[1]
        if target not in seen:
            real_ids.append(target)
            seen.add(target)
        fake_ids.append(f"{target}_{source}")
    return {"real": real_ids, "fake": fake_ids}


# ─────────────────────────────────────────────────────
# PROSES SATU SPLIT
# ─────────────────────────────────────────────────────

def process_split(split_name, split_data, detector, use_mtcnn):
    ids   = get_video_ids(split_data)
    stats = {"real": 0, "fake": 0, "missing": 0}

    for label in ["real", "fake"]:
        out_dir    = os.path.join(OUTPUT_DIR, split_name, label)
        os.makedirs(out_dir, exist_ok=True)
        source_dir = ORIGINAL_DIR if label == "real" else MANIPULATED_DIR

        print(f"\n  [{split_name.upper()}/{label}] {len(ids[label])} video")
        for vid_id in tqdm(ids[label], desc=f"    {split_name}/{label}", ncols=72):
            found = False
            for ext in [".mp4", ".avi", ".mov"]:
                p = os.path.join(source_dir, f"{vid_id}{ext}")
                if os.path.exists(p):
                    found = True; break
            if not found:
                stats["missing"] += 1
                continue
            saved = extract_frames(p, out_dir, vid_id, detector, use_mtcnn)
            stats[label] += saved

    return stats


# ─────────────────────────────────────────────────────
# RINGKASAN
# ─────────────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 62)
    print("RINGKASAN — PROTOKOL RÖSSLER ET AL. 2019")
    print("=" * 62)
    print(f"  {'Split':<8} {'Label':<6} {'Frames':>8} {'Target':>8} {'%':>6}")
    print(f"  {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*6}")

    targets = {"train": 7200, "val": 1400, "test": 1400}
    grand   = 0

    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            folder = os.path.join(OUTPUT_DIR, split, label)
            count  = len([f for f in os.listdir(folder)
                          if f.endswith(".jpg")]) \
                     if os.path.exists(folder) else 0
            tgt    = targets[split]
            pct    = count / tgt * 100 if tgt > 0 else 0
            flag   = "" if pct >= 90 else "  ⚠"
            print(f"  {split:<8} {label:<6} {count:>8,} {tgt:>8,} {pct:>5.1f}%{flag}")
            grand += count
        print(f"  {'':8} {'':6} {'-'*8}")

    print(f"  {'TOTAL':<15} {grand:>8,}")
    print("=" * 62)
    print("\nTarget paper Rössler et al. 2019:")
    print("  7.200 train + 1.400 val + 1.400 test = 10.000 per kelas")
    print("  Total: 20.000 frame (10 frame/video × 2.000 video)")


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("PREPROCESSING — PROTOKOL RÖSSLER ET AL. ICCV 2019")
    print("FF++ C23 Deepfakes | 10 frame/video | 299×299 | MTCNN")
    print("=" * 62)

    # 1. Validasi path
    print("\n[1/5] Validasi path...")
    required = [
        (ORIGINAL_DIR,                           "Original C23"),
        (MANIPULATED_DIR,                        "Manipulated C23"),
        (os.path.join(SPLITS_DIR, "train.json"), "train.json"),
        (os.path.join(SPLITS_DIR, "val.json"),   "val.json"),
        (os.path.join(SPLITS_DIR, "test.json"),  "test.json"),
    ]
    ok = True
    for path, name in required:
        e = os.path.exists(path)
        print(f"  {'OK     ' if e else 'MISSING'} {name}")
        if not e: ok = False
    if not ok:
        print("\n[ERROR] Sesuaikan path di bagian KONFIGURASI")
        sys.exit(1)

    # 2. Hapus data lama
    if os.path.exists(OUTPUT_DIR):
        print(f"\n[2/5] Hapus data lama di '{OUTPUT_DIR}'...")
        shutil.rmtree(OUTPUT_DIR)
        print("  Done.")
    else:
        print(f"\n[2/5] Folder '{OUTPUT_DIR}' belum ada, lanjut.")

    # 3. Setup MTCNN
    print("\n[3/5] Setup face detector...")
    detector, use_mtcnn = setup_mtcnn()

    # 4. Load splits
    print("\n[4/5] Load split resmi FF++...")
    splits = {}
    for name in ["train", "val", "test"]:
        splits[name] = load_split(os.path.join(SPLITS_DIR, f"{name}.json"))
        ids = get_video_ids(splits[name])
        print(f"  {name:<6}: {len(ids['real'])} real · {len(ids['fake'])} fake")

    # 5. Proses
    print(f"\n[5/5] Ekstraksi frame (10 frame/video, protokol paper)...")

    total_saved = 0
    for split_name, split_data in splits.items():
        print(f"\n{'─'*45}")
        print(f"  SPLIT: {split_name.upper()}")
        print(f"{'─'*45}")
        stats = process_split(split_name, split_data, detector, use_mtcnn)
        total_saved += stats["real"] + stats["fake"]

    print(f"\n[SELESAI] Total frame tersimpan: {total_saved:,}")
    print_summary()
    print(f"\nOutput : {os.path.abspath(OUTPUT_DIR)}/")
    print("Lanjut : python src/train.py")


if __name__ == "__main__":
    main()