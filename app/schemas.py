# app/schemas.py
from pydantic import BaseModel
from typing import List, Optional

class Detection(BaseModel):
    cls:        str
    confidence: float
    bbox:       List[float]   # [x1, y1, x2, y2]
    bbox_norm:  List[float]   # normalised [x1, y1, x2, y2]

class ImageResponse(BaseModel):
    model:      str
    detections: List[Detection]
    count:      int
    inference_ms: float

class FrameResult(BaseModel):
    frame:      int
    timestamp:  float
    detections: List[Detection]

class VideoResponse(BaseModel):
    model:               str
    total_frames_sampled: int
    frames:              List[FrameResult]
    processing_ms:       float

class HealthResponse(BaseModel):
    status:     str
    model:      str
    version:    str
    classes:    List[str]
    device:     str