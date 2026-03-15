# app/model.py
import torch
import time
import logging
from pathlib import Path
from ultralytics import YOLO

log = logging.getLogger("uvicorn.error")

CLASS_NAMES = {0: "fully_visible", 1: "partially_visible"}
WEIGHTS_PATH = Path("/app/weights/best.pt")

class Detector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading YOLOv11l from {WEIGHTS_PATH} on {self.device}")
        self.model  = YOLO(str(WEIGHTS_PATH))
        self.model.to(self.device)
        # warm-up pass so first real request isn't slow
        import numpy as np
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False)
        log.info("Model ready")

    def predict_image(self, img_np, conf: float = 0.25):
        t0      = time.perf_counter()
        results = self.model.predict(img_np, conf=conf, verbose=False)[0]
        ms      = round((time.perf_counter() - t0) * 1000, 1)

        detections = []
        h, w = img_np.shape[:2]

        for box, cls_id, conf_score in zip(
            results.boxes.xyxy.cpu().numpy(),
            results.boxes.cls.cpu().numpy().astype(int),
            results.boxes.conf.cpu().numpy(),
        ):
            x1, y1, x2, y2 = box
            detections.append({
                "cls":        CLASS_NAMES.get(cls_id, "unknown"),
                "confidence": round(float(conf_score), 4),
                "bbox":       [round(float(v), 1) for v in [x1, y1, x2, y2]],
                "bbox_norm":  [round(float(x1/w), 4), round(float(y1/h), 4),
                               round(float(x2/w), 4), round(float(y2/h), 4)],
            })
        return detections, ms

# single global instance — loaded once at startup
detector = Detector()