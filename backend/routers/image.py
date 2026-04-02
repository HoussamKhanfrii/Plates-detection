"""
Image inference router.
POST /api/image  – upload an image, get annotated result + plate text.
"""
import logging
import time
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.config import ALLOWED_IMAGE_EXTENSIONS, RESULTS_DIR, UPLOADS_DIR
from backend.database.db import get_db
from backend.models.schemas import ImageInferenceResponse, PlateResult
from backend.services.detector import PlateDetector
from backend.services.history import save_detection
from backend.services.ocr import PlateOCR
from backend.utils.image_utils import draw_detections, read_image

router = APIRouter(prefix="/api/image", tags=["Image Inference"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ImageInferenceResponse)
async def detect_from_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload a JPEG/PNG image.
    Returns bounding boxes, plate text, and confidence scores.
    Also saves an annotated result image.
    """
    # ── Validate file type ────────────────────────────────────────────────────
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_IMAGE_EXTENSIONS}",
        )

    # ── Save upload ───────────────────────────────────────────────────────────
    uid = uuid.uuid4().hex
    upload_path = UPLOADS_DIR / "images" / f"{uid}{suffix}"
    raw_bytes = await file.read()
    upload_path.write_bytes(raw_bytes)

    # ── Run detection ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        image = read_image(upload_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    detector = PlateDetector.get_instance()
    ocr = PlateOCR.get_instance()

    raw_detections = detector.detect(image)

    plates: list[PlateResult] = []
    enriched: list[dict] = []
    for det in raw_detections:
        ocr_result = ocr.read_plate(det["crop"])
        plate = PlateResult(
            plate_text=ocr_result["text"],
            confidence=round(det["confidence"], 4),
            ocr_confidence=ocr_result["ocr_confidence"],
            bbox=det["bbox"],
        )
        plates.append(plate)
        enriched.append(
            {
                "bbox": det["bbox"],
                "plate_text": ocr_result["text"],
                "confidence": det["confidence"],
                "ocr_confidence": ocr_result["ocr_confidence"],
            }
        )

    inference_ms = (time.perf_counter() - t0) * 1000

    # ── Annotate & save result ────────────────────────────────────────────────
    annotated = draw_detections(image, enriched)
    result_filename = f"result_{uid}.jpg"
    result_path = RESULTS_DIR / result_filename
    cv2.imwrite(str(result_path), annotated)

    # ── Persist to DB ─────────────────────────────────────────────────────────
    save_detection(
        db,
        source_type="image",
        filename=file.filename,
        plates=enriched,
        result_path=str(result_path),
    )

    return ImageInferenceResponse(
        filename=file.filename or uid,
        plates=plates,
        plates_count=len(plates),
        inference_time_ms=round(inference_ms, 2),
        result_image_url=f"/results/{result_filename}",
    )
