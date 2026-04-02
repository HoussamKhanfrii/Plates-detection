"""
YOLOv8-based license plate detector service.
Loads the model once at startup and exposes a `detect` method.
Falls back to the pretrained YOLOv8n weights if custom weights are not found.
"""
import logging
from pathlib import Path

import cv2
import numpy as np

from backend.config import (
    YOLO_WEIGHTS,
    YOLO_FALLBACK,
    DETECTION_CONFIDENCE,
    DETECTION_IOU,
)
from backend.utils.image_utils import safe_crop

logger = logging.getLogger(__name__)


class PlateDetector:
    """
    Wraps a YOLOv8 model for license-plate detection.

    Why YOLOv8?
    - Single-stage detector: fast enough for real-time use (≥30 FPS on CPU for 640px)
    - Excellent small-object detection when trained on plate datasets
    - Ultralytics library provides a clean Python API and export options
    - Easy to fine-tune on custom data with minimal configuration
    """

    _instance: "PlateDetector | None" = None

    def __init__(self) -> None:
        self.model = None
        self.model_loaded = False
        self._load_model()

    @classmethod
    def get_instance(cls) -> "PlateDetector":
        """Return the singleton detector (created once per process)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self) -> None:
        """Load YOLO weights; fall back gracefully when custom file is missing."""
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError:
            logger.error("ultralytics not installed – detection disabled")
            return

        weights_path = Path(YOLO_WEIGHTS)
        if weights_path.exists():
            logger.info(f"Loading custom weights: {weights_path}")
            model_path = str(weights_path)
        else:
            logger.warning(
                f"Custom weights not found at {weights_path}. "
                f"Falling back to '{YOLO_FALLBACK}'. "
                "Run training/train.py to produce custom weights."
            )
            model_path = YOLO_FALLBACK

        try:
            self.model = YOLO(model_path)
            self.model_loaded = True
            logger.info(f"Model loaded: {model_path}")
        except Exception as exc:
            logger.error(f"Failed to load YOLO model: {exc}")

    def detect(
        self,
        image: np.ndarray,
        confidence: float | None = None,
        iou: float | None = None,
    ) -> list[dict]:
        """
        Run detection on a BGR numpy image.

        Returns a list of dicts:
            {
                "bbox": [x1, y1, x2, y2],          # pixel coordinates
                "confidence": float,                 # YOLO confidence
                "crop": np.ndarray,                  # cropped plate region
            }
        """
        if not self.model_loaded or self.model is None:
            return []

        conf = confidence if confidence is not None else DETECTION_CONFIDENCE
        iou_thresh = iou if iou is not None else DETECTION_IOU

        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou_thresh,
            verbose=False,
        )

        detections: list[dict] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                x1, y1, x2, y2 = xyxy
                det_conf = float(box.conf[0].cpu().numpy())
                crop = safe_crop(image, x1, y1, x2, y2)
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": det_conf,
                        "crop": crop,
                    }
                )

        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections
