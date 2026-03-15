# app/main.py
import io
import cv2
import time
import numpy as np
import logging

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.model   import detector, CLASS_NAMES
from app.schemas import (
    HealthResponse, ImageResponse, VideoResponse,
    Detection, FrameResult,
)

log = logging.getLogger("uvicorn.error")

app = FastAPI(
    title       = "Paytm Logo Detector",
    description = "YOLOv11-L logo detection API — fully_visible / partially_visible",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ── Health ────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status":  "ok",
        "model":   "yolo11l",
        "version": "1.0.0",
        "classes": list(CLASS_NAMES.values()),
        "device":  detector.device,
    }

# ── Model info ────────────────────────────────────────────────

@app.get("/model/info")
def model_info():
    return {
        "model":        "YOLOv11-L",
        "weights":      "best.pt",
        "classes":      CLASS_NAMES,
        "input_size":   640,
        "parameters":   "25.3M",
        "trained_on":   "Paytm Logo Detection Dataset (754 images)",
        "metrics": {
            "fully_visible":     {"precision": 0.5833, "recall": 0.6053,
                                  "mAP50": 0.6468, "mAP50_95": 0.3610},
            "partially_visible": {"precision": 0.8026, "recall": 0.4082,
                                  "mAP50": 0.5763, "mAP50_95": 0.3194},
            "combined_macro":    {"precision": 0.6930, "recall": 0.5067,
                                  "mAP50": 0.6115, "mAP50_95": 0.3402},
        },
    }

# ── Image inference ───────────────────────────────────────────

@app.post("/predict/image", response_model=ImageResponse)
async def predict_image(
    file: UploadFile = File(...),
    conf: float      = Query(0.25, ge=0.05, le=0.95),
):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Only JPEG/PNG images accepted")

    contents = await file.read()
    try:
        img    = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img)
    except Exception:
        raise HTTPException(400, "Could not decode image")

    detections, ms = detector.predict_image(img_np, conf)

    return {
        "model":        "yolo11l",
        "detections":   detections,
        "count":        len(detections),
        "inference_ms": ms,
    }

# ── Video inference ───────────────────────────────────────────

@app.post("/predict/video", response_model=VideoResponse)
async def predict_video(
    file:         UploadFile = File(...),
    conf:         float      = Query(0.25, ge=0.05, le=0.95),
    sample_every: int        = Query(10,  ge=1,    le=30),
):
    contents = await file.read()
    tmp_path = f"/tmp/{file.filename}"

    with open(tmp_path, "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise HTTPException(400, "Could not open video file")

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_num  = 0
    results    = []
    t0         = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num % sample_every != 0:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections, _ = detector.predict_image(rgb, conf)

        results.append({
            "frame":      frame_num,
            "timestamp":  round(frame_num / fps, 2),
            "detections": detections,
        })

    cap.release()
    processing_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "model":                "yolo11l",
        "total_frames_sampled": len(results),
        "frames":               results,
        "processing_ms":        processing_ms,
    }