"""
SQLAlchemy database setup and session management.
Uses SQLite by default (easy local setup); swap DATABASE_URL for PostgreSQL in production.
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
from backend.config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─── ORM Model ────────────────────────────────────────────────────────────────
class DetectionRecord(Base):
    """Persists every inference result for history tracking."""
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(String(16), nullable=False)          # "image" | "video" | "stream"
    filename = Column(String(256), nullable=True)
    plate_text = Column(String(32), nullable=True)            # OCR result
    confidence = Column(Float, nullable=True)                 # YOLO detection confidence
    ocr_confidence = Column(Float, nullable=True)             # OCR confidence
    result_path = Column(String(512), nullable=True)          # annotated output file
    plates_count = Column(Integer, default=0)
    raw_detections = Column(Text, nullable=True)              # JSON blob of all plates
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


def init_db() -> None:
    """Create all tables if they don't exist yet."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a DB session and ensures it's closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
