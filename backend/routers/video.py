"""
Video inference router.
POST /api/video  – upload a video, process frame by frame, return annotated video + stats.
"""
import logging
import time
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.config import ALLOWED_VIDEO_EXTENSIONS, RESULTS_DIR, UPLOADS_DIR
from backend.database.db import get_db
from backend.models.schemas import VideoInferenceResponse
from backend.services.detector import PlateDetector
from backend.services.history import save_detection
from backend.services.ocr import PlateOCR
from backend.utils.image_utils import draw_detections

router = APIRouter(prefix="/api/video", tags=["Video Inference"])
logger = logging.getLogger(__name__)

# Process every Nth frame to balance speed vs coverage
FRAME_SKIP = 3


@router.post("", response_model=VideoInferenceResponse)
async def detect_from_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a video file (mp4, avi, mov…).
    Returns per-plate stats and a URL to the annotated output video.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported video type '{suffix}'. Allowed: {ALLOWED_VIDEO_EXTENSIONS}",
        )

    uid = uuid.uuid4().hex
    upload_path = UPLOADS_DIR / "videos" / f"{uid}{suffix}"
    raw_bytes = await file.read()
    upload_path.write_bytes(raw_bytes)

    t0 = time.perf_counter()

    cap = cv2.VideoCapture(str(upload_path))
    if not cap.isOpened():
        raise HTTPException(status_code=422, detail="Cannot open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    result_filename = f"result_{uid}.mp4"
    result_path = RESULTS_DIR / result_filename
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(result_path), fourcc, fps, (width, height))

    detector = PlateDetector.get_instance()
    ocr = PlateOCR.get_instance()

    all_plates: list[dict] = []
    unique_texts: set[str] = set()
    frame_idx = 0
    processed_frames = 0
    last_enriched: list[dict] = []  # reuse previous detections on skipped frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            raw_dets = detector.detect(frame)
            enriched: list[dict] = []
            for det in raw_dets:
                ocr_result = ocr.read_plate(det["crop"])
                entry = {
                    "bbox": det["bbox"],
                    "plate_text": ocr_result["text"],
                    "confidence": det["confidence"],
                    "ocr_confidence": ocr_result["ocr_confidence"],
                }
                enriched.append(entry)
                all_plates.append(entry)
                if ocr_result["text"]:
                    unique_texts.add(ocr_result["text"])

            last_enriched = enriched
            processed_frames += 1

        annotated = draw_detections(frame, last_enriched)
        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    inference_ms = (time.perf_counter() - t0) * 1000

    # Persist summary record
    save_detection(
        db,
        source_type="video",
        filename=file.filename,
        plates=all_plates[:50],  # store first 50 to avoid huge JSON blobs
        result_path=str(result_path),
    )

    return VideoInferenceResponse(
        filename=file.filename or uid,
        plates_detected=len(all_plates),
        unique_plates=sorted(unique_texts),
        total_frames=total_frames,
        processed_frames=processed_frames,
        inference_time_ms=round(inference_ms, 2),
        result_video_url=f"/results/{result_filename}",
    )
