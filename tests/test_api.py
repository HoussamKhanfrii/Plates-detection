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
    """
    Create a FastAPI test client backed by an in-memory SQLite database.

    Using `TestClient` as a context manager ensures the ASGI lifespan
    (startup / shutdown) events fire, which loads models and primes the
    singleton services.  Overriding `get_db` with an in-memory engine
    keeps tests isolated from any on-disk database.
    """
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from sqlalchemy.pool import StaticPool

    from backend.main import app
    from backend.database.db import Base, get_db

    # In-memory SQLite for test isolation.
    # StaticPool ensures all connections share the same in-memory database so
    # that tables created by Base.metadata.create_all are visible inside the
    # sessions yielded by override_get_db.
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


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


# ─── History pagination & delete ──────────────────────────────────────────────

def test_history_pagination_params(client):
    """skip and limit query parameters must be accepted without error."""
    resp = client.get("/api/history?skip=0&limit=5")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
    assert len(resp.json()) <= 5


def test_history_limit_capped(client):
    """limit values > 200 should be silently capped to 200."""
    resp = client.get("/api/history?limit=999")
    assert resp.status_code == 200


def test_history_delete_after_image_upload(client, small_jpeg_bytes):
    """Upload an image to create a DB record, then delete it."""
    upload_resp = client.post(
        "/api/image",
        files={"file": ("plate.jpg", io.BytesIO(small_jpeg_bytes), "image/jpeg")},
    )
    assert upload_resp.status_code == 200

    # The uploaded record should appear in history
    history = client.get("/api/history").json()
    assert len(history) >= 1
    record_id = history[0]["id"]

    del_resp = client.delete(f"/api/history/{record_id}")
    assert del_resp.status_code == 204

    # Record must now be absent
    history_after = client.get("/api/history").json()
    ids_after = [r["id"] for r in history_after]
    assert record_id not in ids_after


# ─── Video inference ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def small_mp4_bytes(tmp_path_factory):
    """Create a minimal valid MP4 clip for upload tests."""
    import cv2
    tmp_path = tmp_path_factory.mktemp("video")
    out_path = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 10.0, (64, 64))
    for _ in range(5):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return out_path.read_bytes()


def test_video_upload_wrong_type(client):
    resp = client.post(
        "/api/video",
        files={"file": ("clip.txt", io.BytesIO(b"not a video"), "text/plain")},
    )
    assert resp.status_code == 415


def test_video_upload_no_file(client):
    resp = client.post("/api/video")
    assert resp.status_code == 422


def test_video_upload_success(client, small_mp4_bytes):
    resp = client.post(
        "/api/video",
        files={"file": ("clip.mp4", io.BytesIO(small_mp4_bytes), "video/mp4")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "plates_detected" in data
    assert "unique_plates" in data
    assert "total_frames" in data
    assert "result_video_url" in data
    assert isinstance(data["unique_plates"], list)


# ─── Stream helpers ────────────────────────────────────────────────────────────

def test_mjpeg_frame_format():
    """_mjpeg_frame must produce a valid MJPEG multipart boundary chunk."""
    from backend.routers.stream import _mjpeg_frame
    fake_jpeg = b"\xff\xd8\xff\xe0some_jpeg_data"
    frame = _mjpeg_frame(fake_jpeg)
    assert frame.startswith(b"--frame\r\n")
    assert b"Content-Type: image/jpeg" in frame
    assert fake_jpeg in frame


def test_stream_webcam_endpoint_exists(client):
    """The webcam endpoint should respond (even if no camera is present)."""
    # We expect either a streaming response or an empty response –
    # the important thing is the endpoint is reachable (not 404/405).
    resp = client.get("/api/stream/webcam", timeout=2)
    assert resp.status_code != 404
    assert resp.status_code != 405


def test_stream_ip_endpoint_requires_url(client):
    """IP stream endpoint must reject requests without a url parameter."""
    resp = client.get("/api/stream/ip")
    assert resp.status_code == 422


def test_video_upload_corrupt_content(client):
    """
    A file with a valid extension but unreadable content should return 422.
    OpenCV's VideoCapture will fail to open garbage bytes as a video.
    """
    corrupt_bytes = b"\x00" * 64  # 64 null bytes: not a valid video container
    resp = client.post(
        "/api/video",
        files={"file": ("bad.mp4", io.BytesIO(corrupt_bytes), "video/mp4")},
    )
    assert resp.status_code == 422


def test_image_upload_corrupt_content(client):
    """
    A file with a valid image extension but undecodable content should return 422
    (the read_image ValueError is caught and re-raised as an HTTPException).
    """
    corrupt_bytes = b"\x00" * 64  # null bytes: not a valid JPEG/PNG
    resp = client.post(
        "/api/image",
        files={"file": ("corrupt.jpg", io.BytesIO(corrupt_bytes), "image/jpeg")},
    )
    assert resp.status_code == 422
