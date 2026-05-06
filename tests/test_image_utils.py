"""
Unit tests for backend/utils/image_utils.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.utils.image_utils import (
    draw_detections,
    encode_image_to_bytes,
    preprocess_for_ocr,
    read_image,
    safe_crop,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_bgr(h: int = 100, w: int = 200) -> np.ndarray:
    """Return a random BGR image."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ─── read_image ───────────────────────────────────────────────────────────────

def test_read_image_success(tmp_path):
    img = _make_bgr()
    img_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(img_path), img)
    loaded = read_image(img_path)
    assert loaded.shape == img.shape


def test_read_image_not_found():
    with pytest.raises(ValueError, match="Cannot read image"):
        read_image("/nonexistent/path/to/image.jpg")


def test_read_image_invalid_file(tmp_path):
    bad_path = tmp_path / "notanimage.jpg"
    bad_path.write_bytes(b"this is not image data")
    with pytest.raises(ValueError, match="Cannot read image"):
        read_image(bad_path)


def test_read_image_accepts_string_path(tmp_path):
    img = _make_bgr(50, 50)
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), img)
    loaded = read_image(str(img_path))   # pass as str, not Path
    assert loaded.shape == img.shape


# ─── encode_image_to_bytes ────────────────────────────────────────────────────

def test_encode_jpeg():
    img = _make_bgr()
    data = encode_image_to_bytes(img, ".jpg")
    assert isinstance(data, bytes)
    assert len(data) > 0
    # JPEG magic bytes
    assert data[:2] == b"\xff\xd8"


def test_encode_png():
    img = _make_bgr()
    data = encode_image_to_bytes(img, ".png")
    assert isinstance(data, bytes)
    # PNG magic bytes
    assert data[:4] == b"\x89PNG"


def test_encode_roundtrip(tmp_path):
    """Encode to bytes, decode back – shape must match."""
    img = _make_bgr(60, 80)
    data = encode_image_to_bytes(img, ".png")
    arr = np.frombuffer(data, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert decoded.shape == img.shape


# ─── safe_crop ────────────────────────────────────────────────────────────────

def test_safe_crop_normal():
    img = np.ones((100, 200, 3), dtype=np.uint8)
    crop = safe_crop(img, 10, 10, 50, 60)
    assert crop.shape == (50, 40, 3)


def test_safe_crop_clamped_to_image():
    img = np.ones((100, 100, 3), dtype=np.uint8)
    crop = safe_crop(img, -20, -20, 300, 300)
    assert crop.shape == (100, 100, 3)


def test_safe_crop_negative_coords_become_zero():
    img = np.ones((50, 50, 3), dtype=np.uint8)
    crop = safe_crop(img, -5, -5, 25, 25)
    # Effective window: [0:25, 0:25]
    assert crop.shape == (25, 25, 3)


def test_safe_crop_zero_area():
    """x1==x2 or y1==y2 produces a zero-area (empty) crop without raising."""
    img = np.ones((100, 100, 3), dtype=np.uint8)
    crop = safe_crop(img, 10, 10, 10, 50)  # x1 == x2
    assert crop.size == 0


# ─── preprocess_for_ocr ───────────────────────────────────────────────────────

def test_preprocess_small_image_is_upscaled():
    """Images narrower than 120 px should be upscaled."""
    small = _make_bgr(30, 50)  # width 50 < 120
    result = preprocess_for_ocr(small)
    assert result.ndim == 2            # grayscale
    assert result.shape[1] >= 120      # upscaled width


def test_preprocess_wide_image_not_upscaled():
    """Images already wider than 120 px must not be upscaled."""
    wide = _make_bgr(60, 200)          # width 200 >= 120
    result = preprocess_for_ocr(wide)
    assert result.ndim == 2
    assert result.shape[1] == 200      # width preserved


def test_preprocess_output_is_binary():
    """Output should contain only 0 and 255 (adaptive threshold result)."""
    img = _make_bgr(40, 150)
    result = preprocess_for_ocr(img)
    unique_vals = set(result.flatten().tolist())
    assert unique_vals.issubset({0, 255})


# ─── draw_detections ──────────────────────────────────────────────────────────

def test_draw_detections_returns_copy():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    plates = [{"bbox": [10, 10, 100, 50], "plate_text": "ABC123", "confidence": 0.9}]
    annotated = draw_detections(img, plates)
    assert annotated.shape == img.shape
    # Original image must not be mutated
    assert np.all(img == 0)


def test_draw_detections_empty_list():
    img = _make_bgr()
    original = img.copy()
    annotated = draw_detections(img, [])
    assert np.array_equal(annotated, original)


def test_draw_detections_missing_optional_keys():
    """bbox is required; plate_text and confidence are optional (have defaults)."""
    img = _make_bgr()
    plates = [{"bbox": [5, 5, 80, 40]}]  # no plate_text / confidence
    annotated = draw_detections(img, plates)
    assert annotated.shape == img.shape


def test_draw_detections_multiple_plates():
    img = _make_bgr(300, 500)
    plates = [
        {"bbox": [10, 10, 100, 50], "plate_text": "PLATE1", "confidence": 0.95},
        {"bbox": [200, 100, 350, 160], "plate_text": "PLATE2", "confidence": 0.80},
    ]
    annotated = draw_detections(img, plates)
    assert annotated.shape == img.shape


def test_encode_image_unsupported_extension():
    """encode_image_to_bytes raises RuntimeError when cv2.imencode reports failure."""
    from unittest.mock import patch
    img = _make_bgr()
    # Simulate imencode returning success=False (e.g. unsupported codec)
    with patch("backend.utils.image_utils.cv2.imencode", return_value=(False, None)):
        with pytest.raises(RuntimeError, match="Failed to encode"):
            encode_image_to_bytes(img, ".jpg")
