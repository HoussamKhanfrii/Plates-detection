"""
Unit tests for backend/services/ – history, OCR, and detector services.
Uses an isolated in-memory SQLite database for history tests.
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.database.db import Base, DetectionRecord
from backend.services.history import delete_record, get_history, save_detection


# ─── DB fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture()
def db_session():
    """Provide a fresh in-memory SQLite session for each test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


# ─── save_detection ───────────────────────────────────────────────────────────

def test_save_detection_no_plates(db_session):
    record = save_detection(db_session, "image", "photo.jpg", [])
    assert record.id is not None
    assert record.source_type == "image"
    assert record.filename == "photo.jpg"
    assert record.plates_count == 0
    assert record.plate_text is None


def test_save_detection_with_plates(db_session):
    plates = [
        {"plate_text": "ABC123", "confidence": 0.95, "ocr_confidence": 0.88, "bbox": [0, 0, 100, 50]},
        {"plate_text": "XYZ789", "confidence": 0.72, "ocr_confidence": 0.65, "bbox": [200, 0, 300, 50]},
    ]
    record = save_detection(db_session, "image", "img.jpg", plates, result_path="/tmp/result.jpg")
    assert record.plate_text == "ABC123"   # first plate = primary
    assert record.confidence == pytest.approx(0.95)
    assert record.plates_count == 2
    assert record.result_path == "/tmp/result.jpg"
    raw = json.loads(record.raw_detections)
    assert len(raw) == 2
    # Crop key must be excluded from the JSON blob
    assert all("crop" not in entry for entry in raw)


def test_save_detection_crop_excluded(db_session):
    """The numpy crop array must not be serialised into raw_detections."""
    plates = [
        {
            "plate_text": "AA111",
            "confidence": 0.8,
            "ocr_confidence": 0.7,
            "bbox": [0, 0, 50, 30],
            "crop": np.zeros((30, 50, 3), dtype=np.uint8),
        }
    ]
    record = save_detection(db_session, "image", "img.jpg", plates)
    raw = json.loads(record.raw_detections)
    assert "crop" not in raw[0]


def test_save_detection_video_source(db_session):
    record = save_detection(db_session, "video", "clip.mp4", [])
    assert record.source_type == "video"


# ─── get_history ──────────────────────────────────────────────────────────────

def test_get_history_empty(db_session):
    assert get_history(db_session) == []


def test_get_history_returns_records(db_session):
    save_detection(db_session, "image", "a.jpg", [])
    save_detection(db_session, "image", "b.jpg", [])
    records = get_history(db_session)
    assert len(records) == 2


def test_get_history_newest_first(db_session):
    save_detection(db_session, "image", "first.jpg", [])
    save_detection(db_session, "image", "second.jpg", [])
    records = get_history(db_session)
    assert records[0].filename == "second.jpg"


def test_get_history_limit(db_session):
    for i in range(5):
        save_detection(db_session, "image", f"{i}.jpg", [])
    records = get_history(db_session, limit=3)
    assert len(records) == 3


def test_get_history_skip(db_session):
    for i in range(4):
        save_detection(db_session, "image", f"{i}.jpg", [])
    all_records = get_history(db_session)
    skipped = get_history(db_session, skip=2)
    assert len(skipped) == len(all_records) - 2


# ─── delete_record ────────────────────────────────────────────────────────────

def test_delete_record_success(db_session):
    record = save_detection(db_session, "image", "del.jpg", [])
    result = delete_record(db_session, record.id)
    assert result is True
    assert get_history(db_session) == []


def test_delete_record_not_found(db_session):
    result = delete_record(db_session, 99999)
    assert result is False


def test_delete_record_idempotent(db_session):
    record = save_detection(db_session, "image", "del.jpg", [])
    record_id = record.id
    delete_record(db_session, record_id)
    # Second delete should return False gracefully
    result = delete_record(db_session, record_id)
    assert result is False


# ─── PlateOCR._clean_plate_text ───────────────────────────────────────────────

class TestCleanPlateText:
    """Tests for the static text-cleaning method in PlateOCR."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from backend.services.ocr import PlateOCR
        self.clean = PlateOCR._clean_plate_text

    def test_uppercase(self):
        assert self.clean("ab123") == "AB123"

    def test_strips_whitespace(self):
        assert self.clean("  AB 123  ") == "AB 123"

    def test_removes_noise_chars(self):
        # Characters outside [A-Z0-9\-\s] should be removed
        assert self.clean("AB@123!") == "AB123"

    def test_collapses_spaces(self):
        assert self.clean("AB  123") == "AB 123"

    def test_collapses_repeated_hyphens(self):
        assert self.clean("AB--123") == "AB-123"

    def test_empty_string(self):
        assert self.clean("") == ""

    def test_only_noise_chars(self):
        assert self.clean("@@@!!!") == ""

    def test_mixed_valid_and_noise(self):
        result = self.clean("A1-B2 C3#D4")
        assert "#" not in result
        assert "A1-B2 C3D4" == result


# ─── PlateOCR.read_plate when disabled ────────────────────────────────────────

def test_read_plate_when_ocr_disabled():
    """When ready=False or reader=None, read_plate must return empty results."""
    from backend.services.ocr import PlateOCR

    ocr = PlateOCR.__new__(PlateOCR)
    ocr.reader = None
    ocr.ready = False

    crop = np.zeros((30, 100, 3), dtype=np.uint8)
    result = ocr.read_plate(crop)
    assert result["text"] == ""
    assert result["raw_text"] == ""
    assert result["ocr_confidence"] == 0.0


def test_read_plate_empty_crop():
    """A zero-size crop must return empty results without raising."""
    from backend.services.ocr import PlateOCR

    ocr = PlateOCR.__new__(PlateOCR)
    ocr.reader = MagicMock()
    ocr.ready = True

    empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    result = ocr.read_plate(empty_crop)
    assert result["text"] == ""
    # reader.readtext should never have been called for an empty crop
    ocr.reader.readtext.assert_not_called()


# ─── PlateDetector.detect when model not loaded ───────────────────────────────

def test_detector_detect_no_model():
    """detect() must return an empty list when the model failed to load."""
    from backend.services.detector import PlateDetector

    detector = PlateDetector.__new__(PlateDetector)
    detector.model = None
    detector.model_loaded = False

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    results = detector.detect(image)
    assert results == []


def test_detector_detect_sorts_by_confidence():
    """detect() must return detections sorted by confidence descending."""
    from backend.services.detector import PlateDetector

    detector = PlateDetector.__new__(PlateDetector)
    detector.model_loaded = True

    # Build a fake YOLO result with two boxes
    low_conf_box = MagicMock()
    low_conf_box.xyxy = [MagicMock()]
    low_conf_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([0, 0, 50, 30], dtype=float)
    low_conf_box.conf = [MagicMock()]
    low_conf_box.conf[0].cpu.return_value.numpy.return_value = np.float32(0.5)

    high_conf_box = MagicMock()
    high_conf_box.xyxy = [MagicMock()]
    high_conf_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([0, 0, 50, 30], dtype=float)
    high_conf_box.conf = [MagicMock()]
    high_conf_box.conf[0].cpu.return_value.numpy.return_value = np.float32(0.9)

    fake_result = MagicMock()
    fake_result.boxes = [low_conf_box, high_conf_box]

    detector.model = MagicMock()
    detector.model.predict.return_value = [fake_result]

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    dets = detector.detect(image)

    assert len(dets) == 2
    assert dets[0]["confidence"] >= dets[1]["confidence"]


def test_detector_detect_empty_boxes():
    """detect() must handle a result with boxes=None gracefully."""
    from backend.services.detector import PlateDetector

    detector = PlateDetector.__new__(PlateDetector)
    detector.model_loaded = True

    fake_result = MagicMock()
    fake_result.boxes = None

    detector.model = MagicMock()
    detector.model.predict.return_value = [fake_result]

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    dets = detector.detect(image)
    assert dets == []


# ─── PlateOCR.read_plate – successful OCR path ────────────────────────────────

def test_read_plate_with_ocr_results():
    """
    When reader.readtext returns recognised text, read_plate must aggregate
    results and clean the text.
    """
    from backend.services.ocr import PlateOCR

    ocr = PlateOCR.__new__(PlateOCR)
    ocr.ready = True
    ocr.reader = MagicMock()
    # EasyOCR result format: [(bbox, text, confidence), ...]
    ocr.reader.readtext.return_value = [
        (None, "AB 123", 0.92),
        (None, "CD", 0.85),
    ]

    crop = np.ones((30, 100, 3), dtype=np.uint8)
    result = ocr.read_plate(crop)

    assert result["text"] != ""
    assert result["raw_text"] == "AB 123 CD"
    assert result["ocr_confidence"] == pytest.approx((0.92 + 0.85) / 2, abs=1e-3)


def test_read_plate_readtext_returns_empty():
    """When readtext returns an empty list, read_plate must return empty results."""
    from backend.services.ocr import PlateOCR

    ocr = PlateOCR.__new__(PlateOCR)
    ocr.ready = True
    ocr.reader = MagicMock()
    ocr.reader.readtext.return_value = []

    crop = np.ones((30, 100, 3), dtype=np.uint8)
    result = ocr.read_plate(crop)
    assert result["text"] == ""
    assert result["ocr_confidence"] == 0.0


def test_read_plate_readtext_raises():
    """When readtext raises, read_plate must catch the exception and return empty."""
    from backend.services.ocr import PlateOCR

    ocr = PlateOCR.__new__(PlateOCR)
    ocr.ready = True
    ocr.reader = MagicMock()
    ocr.reader.readtext.side_effect = RuntimeError("simulated OCR failure")

    crop = np.ones((30, 100, 3), dtype=np.uint8)
    result = ocr.read_plate(crop)
    assert result["text"] == ""
    assert result["ocr_confidence"] == 0.0
