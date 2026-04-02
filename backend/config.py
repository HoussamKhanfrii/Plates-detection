"""
Central configuration for the backend.
Settings are read from environment variables with sensible defaults.
"""
import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models" / "weights"
UPLOADS_DIR = ROOT_DIR / "uploads"
RESULTS_DIR = UPLOADS_DIR / "results"
DB_PATH = ROOT_DIR / "plates_history.db"

# Ensure runtime directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
(UPLOADS_DIR / "images").mkdir(exist_ok=True)
(UPLOADS_DIR / "videos").mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Model ───────────────────────────────────────────────────────────────────
# Path to the trained YOLOv8 weights. Falls back to the nano pretrained model
# so the server can start even without custom training.
YOLO_WEIGHTS: str = os.getenv(
    "YOLO_WEIGHTS",
    str(MODELS_DIR / "best.pt"),
)
YOLO_FALLBACK: str = "yolov8n.pt"          # used when custom weights not found
DETECTION_CONFIDENCE: float = float(os.getenv("DETECTION_CONFIDENCE", "0.4"))
DETECTION_IOU: float = float(os.getenv("DETECTION_IOU", "0.45"))

# ─── OCR ─────────────────────────────────────────────────────────────────────
OCR_GPU: bool = os.getenv("OCR_GPU", "false").lower() == "true"
OCR_LANGUAGES: list[str] = ["en"]          # extend for multi-language plates

# ─── API / Server ─────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ─── Database ─────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# ─── CORS ─────────────────────────────────────────────────────────────────────
CORS_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:5500"
).split(",")
