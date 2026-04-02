"""
API integration tests using FastAPI's TestClient.
Run with: pytest tests/ -v
"""
import io
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """Create a FastAPI test client."""
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


@pytest.fixture
def small_jpeg_bytes():
    """Create a minimal valid JPEG image as bytes for upload tests."""
    import cv2
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    # Draw a fake plate-like rectangle
    cv2.rectangle(img, (20, 30), (180, 70), (255, 255, 255), -1)
    cv2.putText(img, "AB123CD", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ─── Health ───────────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "ocr_ready" in data


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


# ─── Image inference ──────────────────────────────────────────────────────────

def test_image_upload_success(client, small_jpeg_bytes):
    resp = client.post(
        "/api/image",
        files={"file": ("test_plate.jpg", io.BytesIO(small_jpeg_bytes), "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "plates" in data
    assert "inference_time_ms" in data
    assert "result_image_url" in data
    assert isinstance(data["plates"], list)


def test_image_upload_wrong_type(client):
    resp = client.post(
        "/api/image",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
    )
    assert resp.status_code == 415


def test_image_upload_no_file(client):
    resp = client.post("/api/image")
    assert resp.status_code == 422


# ─── History ──────────────────────────────────────────────────────────────────

def test_history_returns_list(client):
    resp = client.get("/api/history")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_history_delete_nonexistent(client):
    resp = client.delete("/api/history/999999")
    assert resp.status_code == 404


# ─── Plate text utility ───────────────────────────────────────────────────────

def test_normalize_plate():
    from utils.plate_utils import normalize_plate, is_valid_plate
    assert normalize_plate("ab 123 cd!") == "AB 123 CD"
    assert normalize_plate("") == ""
    assert is_valid_plate("AB123") is True
    assert is_valid_plate("X") is False
    assert is_valid_plate("A" * 20) is False


# ─── Image utility ────────────────────────────────────────────────────────────

def test_preprocess_for_ocr():
    from backend.utils.image_utils import preprocess_for_ocr
    img = np.random.randint(0, 255, (40, 100, 3), dtype=np.uint8)
    result = preprocess_for_ocr(img)
    # Result should be grayscale (2D) after thresholding
    assert result.ndim == 2
    assert result.shape[1] >= 120  # upscaled to min width


def test_draw_detections():
    from backend.utils.image_utils import draw_detections
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    plates = [{"bbox": [10, 10, 100, 50], "plate_text": "TEST123", "confidence": 0.9}]
    annotated = draw_detections(img, plates)
    assert annotated.shape == img.shape


def test_safe_crop():
    from backend.utils.image_utils import safe_crop
    img = np.ones((100, 100, 3), dtype=np.uint8)
    crop = safe_crop(img, -10, -10, 200, 200)   # out-of-bounds coords
    assert crop.shape == (100, 100, 3)           # clamped to image size
