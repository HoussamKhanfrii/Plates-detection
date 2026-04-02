"""
FastAPI application entry point.

Run with:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import CORS_ORIGINS, RESULTS_DIR
from backend.database.db import init_db
from backend.models.schemas import HealthResponse
from backend.routers import history, image, stream, video
from backend.services.detector import PlateDetector
from backend.services.ocr import PlateOCR

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="License Plate Detection API",
    description=(
        "End-to-end automatic license plate detection and recognition. "
        "Supports image upload, video processing, and real-time webcam streaming."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],   # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static file serving ──────────────────────────────────────────────────────
# Serve annotated result files at /results/<filename>
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

# Serve the frontend from /frontend (for the built-in HTML/JS UI)
_frontend_dir = Path(__file__).parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(image.router)
app.include_router(video.router)
app.include_router(stream.router)
app.include_router(history.router)


# ─── Lifecycle ────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info("Initialising database…")
    init_db()
    logger.info("Pre-loading detector and OCR models…")
    PlateDetector.get_instance()
    PlateOCR.get_instance()
    logger.info("Server ready.")


# ─── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    detector = PlateDetector.get_instance()
    ocr = PlateOCR.get_instance()
    return HealthResponse(
        status="ok",
        model_loaded=detector.model_loaded,
        ocr_ready=ocr.ready,
    )


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "License Plate Detection API is running.",
        "docs": "/docs",
        "ui": "/ui",
    }
