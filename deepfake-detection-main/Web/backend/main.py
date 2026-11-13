"""
main.py — FastAPI Backend
==========================
Update dari versi sebelumnya:
  - Model info sesuai protokol Rössler et al. 2019
  - MODELS_DIR disesuaikan ke lokasi aktual
  - Validasi content-type lebih fleksibel
"""

import os
import uuid
import shutil
import logging
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from detector import DeepfakeDetector, MODEL_CONFIG

app = FastAPI(title="Deepfake Detection API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000",
                   "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files untuk Grad-CAM
os.makedirs("static/gradcam", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Sesuaikan path models ──────────────────────────────
# Ganti path ini sesuai lokasi folder models di komputer kamu
MODELS_DIR = "../models"   # relatif dari folder backend/
# Jika tidak ketemu, coba path absolut:
# MODELS_DIR = r"C:\Users\adama\OneDrive\Desktop\deepfake_detection\models"
# ──────────────────────────────────────────────────────

print(f"Loading models dari: {os.path.abspath(MODELS_DIR)}")
detector = DeepfakeDetector(models_dir=MODELS_DIR)


@app.get("/health")
def health():
    loaded = detector.get_loaded_models()
    return {
        "status": "ok",
        "models_loaded": loaded,
        "total": len(loaded),
    }


@app.get("/models")
def get_models():
    """Return daftar model dengan info lengkap."""
    model_info = []
    for model_id, cfg in MODEL_CONFIG.items():
        loaded = model_id in detector.models
        model_info.append({
            "id":          model_id,
            "name":        cfg["name"],
            "desc":        _get_desc(model_id),
            "params":      cfg["params"],
            "input_size":  f"{cfg['input_size'][0]}×{cfg['input_size'][1]}",
            "available":   loaded,
        })
    return {"models": model_info}


def _get_desc(model_id):
    descs = {
        "xception":    "Depthwise separable conv · Best accuracy · 96.2%",
        "resnet50":    "Deep residual network · 50 layers · 90.3%",
        "mobilenetv2": "Inverted residuals · Fastest & efficient · 91.3%",
        "mesonet4":    "Purpose-built deepfake detector · 83.2%",
    }
    return descs.get(model_id, "")


@app.post("/detect")
async def detect(
    video: UploadFile = File(...),
    model_id: str     = Form(...),
):
    # Validasi model
    valid_models = list(MODEL_CONFIG.keys())
    if model_id not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model tidak valid: {model_id}. "
                   f"Pilihan: {valid_models}"
        )

    if model_id not in detector.models:
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model_id}' belum ter-load. "
                   f"Pastikan file .keras tersedia."
        )

    # Validasi file video
    content_type = video.content_type or ""
    allowed = ["video/mp4", "video/avi", "video/quicktime",
               "video/x-msvideo", "video/x-matroska",
               "application/octet-stream"]
    filename_lower = (video.filename or "").lower()
    valid_ext = any(filename_lower.endswith(ext)
                    for ext in [".mp4", ".avi", ".mov", ".mkv"])

    if content_type not in allowed and not valid_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Format tidak didukung: {content_type}"
        )

    # Simpan video sementara
    ext      = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    tmp_path = f"static/gradcam/tmp_{uuid.uuid4().hex}{ext}"

    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    file_size = os.path.getsize(tmp_path)

    try:
        result = detector.detect(
            video_path=tmp_path,
            model_id=model_id,
        )
        result["video_info"] = {
            "filename":   video.filename,
            "size_bytes": file_size,
            "size_kb":    round(file_size / 1024, 1),
            "model_used": model_id,
            "model_name": MODEL_CONFIG[model_id]["name"],
            "input_size": f"{MODEL_CONFIG[model_id]['input_size'][0]}×"
                          f"{MODEL_CONFIG[model_id]['input_size'][1]}",
        }
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)