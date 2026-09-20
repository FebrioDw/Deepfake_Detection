"""
preprocess.py — Data Extraction Pipeline
=============================================================
Implementation of the official FaceForensics++ extraction protocol 
(Rössler et al., ICCV 2019):

  - Uniform sampling: 10 frames per video
  - Face detection & cropping via MTCNN (fallback to Haar Cascades)
  - Official FF++ split: 720 train / 140 val / 140 test
  - Image target size: 299x299 (Optimized for Xception/ResNet/MobileNet)
"""

import os
import sys
import json
import shutil
import logging
import warnings

# Suppress verbose TensorFlow warnings for cleaner console output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import cv2
import numpy as np
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
ORIGINAL_DIR    = "dataset/Original_C23"
MANIPULATED_DIR = "dataset/Manipulated_C23"
SPLITS_DIR      = "dataset/splits"
OUTPUT_DIR      = "dataset_split"  # Aligned with training scripts

IMG_SIZE         = (299, 299)   
FRAMES_PER_VIDEO = 10           
FACE_MARGIN      = 0.3          
MIN_FACE_SIZE    = 20
MTCNN_THRESHOLD  = [0.5, 0.6, 0.7]

# ==========================================
# FACE DETECTOR INITIALIZATION
# ==========================================
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
        print(f"[INFO] MTCNN successfully initialized on {device.upper()}")
        return mtcnn, True
    except ImportError:
        print("[WARNING] MTCNN initialization failed. Falling back to Haar Cascades.")
        haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        return haar, False

# ==========================================
# FACE CROPPING MODULE
# ==========================================
def crop_face_mtcnn(frame_bgr, mtcnn):
    from PIL import Image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    boxes, probs = mtcnn.detect(pil)
    
    if boxes is None or len(boxes) == 0:
        return None
        
    idx = int(np.argmax(probs)) if probs is not None else 0
    x1, y1, x2, y2 = boxes[idx]
    h, w = frame_bgr.shape[:2]
    
    mx = int((x2 - x1) * FACE_MARGIN)
    my = int((y2 - y1) * FACE_MARGIN)
    
    x1, y1 = max(0, int(x1) - mx), max(0, int(y1) - my)
    x2, y2 = min(w, int(x2) + mx), min(h, int(y2) + my)
    
    crop = frame_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, IMG_SIZE) if crop.size > 0 else None

def crop_face_haar(frame_bgr, haar):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )
    
    if len(faces) == 0:
        return None
        
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    H, W = frame_bgr.shape[:2]
    
    mx, my = int(w * FACE_MARGIN), int(h * FACE_MARGIN)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(W, x + w + mx), min(H, y + h + my)
    
    crop = frame_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, IMG_SIZE) if crop.size > 0 else None

def detect_face(frame, detector, use_mtcnn):
    return crop_face_mtcnn(frame, detector) if use_mtcnn else crop_face_haar(frame, detector)

# ==========================================
# FRAME EXTRACTION
# ==========================================
def extract_frames(video_path, out_dir, video_id, detector, use_mtcnn):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return 0

    num_samples = min(FRAMES_PER_VIDEO, total_frames)
    indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
    saved_frames = 0

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
            
        face = detect_face(frame, detector, use_mtcnn)
        if face is None:
            continue
            
        filename = f"{video_id}_f{int(idx):05d}.jpg"
        filepath = os.path.join(out_dir, filename)
        cv2.imwrite(filepath, face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_frames += 1

    cap.release()
    return saved_frames

# ==========================================
# DATASET SPLIT MANAGEMENT
# ==========================================
def load_split(json_path):
    with open(json_path) as f:
        return json.load(f)

def get_video_ids(split_data):
    real_ids, fake_ids = [], []
    seen = set()
    for pair in split_data:
        target, source = pair[0], pair[1]
        if target not in seen:
            real_ids.append(target)
            seen.add(target)
        fake_ids.append(f"{target}_{source}")
    return {"real": real_ids, "fake": fake_ids}

def process_split(split_name, split_data, detector, use_mtcnn):
    ids = get_video_ids(split_data)
    stats = {"real": 0, "fake": 0, "missing": 0}

    for label in ["real", "fake"]:
        out_dir = os.path.join(OUTPUT_DIR, split_name, label)
        os.makedirs(out_dir, exist_ok=True)
        source_dir = ORIGINAL_DIR if label == "real" else MANIPULATED_DIR

        print(f"\n[PROCESS] {split_name.upper()} / {label.upper()} ({len(ids[label])} videos)")
        for vid_id in tqdm(ids[label], desc=f"Extracting", ncols=72):
            found = False
            for ext in [".mp4", ".avi", ".mov"]:
                video_path = os.path.join(source_dir, f"{vid_id}{ext}")
                if os.path.exists(video_path):
                    found = True
                    break
            
            if not found:
                stats["missing"] += 1
                continue
                
            saved_count = extract_frames(video_path, out_dir, vid_id, detector, use_mtcnn)
            stats[label] += saved_count

    return stats

# ==========================================
# EXECUTION & SUMMARY
# ==========================================
def print_summary():
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY (RÖSSLER ET AL. 2019 PROTOCOL)")
    print("=" * 60)
    print(f"{'Split':<10} {'Label':<8} {'Extracted':>12} {'Target':>10} {'Coverage'}")
    print("-" * 60)

    targets = {"train": 7200, "val": 1400, "test": 1400}
    grand_total = 0

    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            folder = os.path.join(OUTPUT_DIR, split, label)
            count = len([f for f in os.listdir(folder) if f.endswith(".jpg")]) if os.path.exists(folder) else 0
            tgt = targets[split]
            pct = (count / tgt * 100) if tgt > 0 else 0
            
            print(f"{split:<10} {label:<8} {count:>12,} {tgt:>10,} {pct:>7.1f}%")
            grand_total += count
        print("-" * 60)

    print(f"{'TOTAL GENERATED FRAMES':<20} {grand_total:>12,}")
    print("=" * 60)

def main():
    print("============================================================")
    print(" INITIALIZING DEEPFAKE PREPROCESSING PIPELINE")
    print("============================================================")

    # 1. Path Validation
    required_paths = [
        (ORIGINAL_DIR, "Original Dataset"),
        (MANIPULATED_DIR, "Manipulated Dataset"),
        (os.path.join(SPLITS_DIR, "train.json"), "Train Split JSON"),
        (os.path.join(SPLITS_DIR, "val.json"), "Validation Split JSON"),
        (os.path.join(SPLITS_DIR, "test.json"), "Test Split JSON"),
    ]
    
    is_valid = True
    for path, name in required_paths:
        exists = os.path.exists(path)
        print(f"[{'OK' if exists else 'FAIL'}] {name}")
        if not exists: is_valid = False
        
    if not is_valid:
        print("\n[ERROR] Missing required paths. Please configure dataset directories.")
        sys.exit(1)

    # 2. Workspace Preparation
    if os.path.exists(OUTPUT_DIR):
        print(f"\n[INFO] Cleaning previous output directory: {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    
    # 3. Model Setup
    print("\n[INFO] Setting up extraction model...")
    detector, use_mtcnn = setup_mtcnn()

    # 4. Load Splits
    splits = {name: load_split(os.path.join(SPLITS_DIR, f"{name}.json")) for name in ["train", "val", "test"]}

    # 5. Extraction Process
    total_saved = 0
    for split_name, split_data in splits.items():
        stats = process_split(split_name, split_data, detector, use_mtcnn)
        total_saved += stats["real"] + stats["fake"]

    # 6. Final Report
    print_summary()
    print(f"\n[SUCCESS] Pipeline completed. Ready for model training.")

if __name__ == "__main__":
    main()