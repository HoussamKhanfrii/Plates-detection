"""
History service: CRUD operations around the DetectionRecord ORM model.
"""
import json
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from backend.database.db import DetectionRecord

logger = logging.getLogger(__name__)


def save_detection(
    db: Session,
    source_type: str,
    filename: str | None,
    plates: list[dict],
    result_path: str | None = None,
) -> DetectionRecord:
    """
    Persist a detection event.

    plates: list of {plate_text, confidence, ocr_confidence, bbox}
    """
    # Pick the highest-confidence plate as the "primary" result
    primary = plates[0] if plates else {}
    record = DetectionRecord(
        source_type=source_type,
        filename=filename,
        plate_text=primary.get("plate_text"),
        confidence=primary.get("confidence"),
        ocr_confidence=primary.get("ocr_confidence"),
        result_path=result_path,
        plates_count=len(plates),
        raw_detections=json.dumps(
            [{k: v for k, v in p.items() if k != "crop"} for p in plates]
        ),
        created_at=datetime.now(timezone.utc),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_history(db: Session, skip: int = 0, limit: int = 50) -> list[DetectionRecord]:
    return (
        db.query(DetectionRecord)
        .order_by(DetectionRecord.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def delete_record(db: Session, record_id: int) -> bool:
    record = db.query(DetectionRecord).filter(DetectionRecord.id == record_id).first()
    if record is None:
        return False
    db.delete(record)
    db.commit()
    return True
