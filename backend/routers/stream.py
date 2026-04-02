"""
Real-time webcam/IP-camera stream router.
GET /api/stream/webcam  – MJPEG stream from the default webcam (index 0).
GET /api/stream/ip?url=<rtsp_or_http_url>  – MJPEG stream from an IP camera.

The server captures frames, runs detection + OCR, annotates them, and
streams the result as multipart/x-mixed-replace (MJPEG).
"""
import logging
import time

import cv2
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from backend.services.detector import PlateDetector
from backend.services.ocr import PlateOCR
from backend.utils.image_utils import draw_detections, encode_image_to_bytes

router = APIRouter(prefix="/api/stream", tags=["Real-time Stream"])
logger = logging.getLogger(__name__)

# Run detection every N frames to maintain smooth frame rate on CPU
DETECT_EVERY_N = 5


def _mjpeg_frame(jpeg_bytes: bytes) -> bytes:
    """Wrap JPEG bytes in an MJPEG multipart boundary."""
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
    )


def _generate_stream(source: int | str):
    """
    Generator that captures frames from `source`, annotates detected plates,
    and yields MJPEG frames.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        return

    detector = PlateDetector.get_instance()
    ocr = PlateOCR.get_instance()

    frame_idx = 0
    last_enriched: list[dict] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Try to reconnect for IP cameras
                time.sleep(0.1)
                cap.release()
                cap = cv2.VideoCapture(source)
                continue

            if frame_idx % DETECT_EVERY_N == 0:
                raw_dets = detector.detect(frame)
                enriched: list[dict] = []
                for det in raw_dets:
                    ocr_result = ocr.read_plate(det["crop"])
                    enriched.append(
                        {
                            "bbox": det["bbox"],
                            "plate_text": ocr_result["text"],
                            "confidence": det["confidence"],
                            "ocr_confidence": ocr_result["ocr_confidence"],
                        }
                    )
                last_enriched = enriched

            annotated = draw_detections(frame, last_enriched)
            jpeg_bytes = encode_image_to_bytes(annotated, ".jpg")
            yield _mjpeg_frame(jpeg_bytes)
            frame_idx += 1
    finally:
        cap.release()


@router.get("/webcam")
def webcam_stream(camera_index: int = Query(0, description="OpenCV camera index")):
    """Stream annotated video from the local webcam (MJPEG)."""
    return StreamingResponse(
        _generate_stream(camera_index),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/ip")
def ip_camera_stream(url: str = Query(..., description="RTSP or HTTP camera URL")):
    """Stream annotated video from an IP / RTSP camera (MJPEG)."""
    return StreamingResponse(
        _generate_stream(url),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
