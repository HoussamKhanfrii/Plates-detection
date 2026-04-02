"""
Detection history router.
GET  /api/history         – paginated list of past detections
DELETE /api/history/{id}  – remove a specific record
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.db import get_db
from backend.models.schemas import DetectionHistoryItem
from backend.services.history import delete_record, get_history

router = APIRouter(prefix="/api/history", tags=["History"])


@router.get("", response_model=list[DetectionHistoryItem])
def list_history(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Return the most recent detection records (newest first)."""
    return get_history(db, skip=skip, limit=min(limit, 200))


@router.delete("/{record_id}", status_code=204)
def remove_record(record_id: int, db: Session = Depends(get_db)):
    """Delete a single detection record by ID."""
    if not delete_record(db, record_id):
        raise HTTPException(status_code=404, detail="Record not found")
