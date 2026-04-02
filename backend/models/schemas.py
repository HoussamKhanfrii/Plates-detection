"""
Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class PlateResult(BaseModel):
    """Single plate detection result."""
    plate_text: str = Field(..., description="OCR-extracted plate text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="YOLO detection confidence")
    ocr_confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score")
    bbox: list[int] = Field(..., description="[x1, y1, x2, y2] bounding box in pixels")


class ImageInferenceResponse(BaseModel):
    source_type: str = "image"
    filename: str
    plates: list[PlateResult]
    plates_count: int
    inference_time_ms: float
    result_image_url: str


class VideoInferenceResponse(BaseModel):
    source_type: str = "video"
    filename: str
    plates_detected: int
    unique_plates: list[str]
    total_frames: int
    processed_frames: int
    inference_time_ms: float
    result_video_url: str


class DetectionHistoryItem(BaseModel):
    id: int
    source_type: str
    filename: Optional[str]
    plate_text: Optional[str]
    confidence: Optional[float]
    ocr_confidence: Optional[float]
    result_path: Optional[str]
    plates_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    ocr_ready: bool
    version: str = "1.0.0"
