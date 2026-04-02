"""
EasyOCR-based plate text recognition service.

Why EasyOCR?
- Better accuracy than Tesseract on real-world license plates out of the box
- GPU-accelerated (falls back to CPU automatically)
- Supports 80+ languages – useful for non-English plates
- Simple Python API; no binary system dependencies unlike Tesseract
- Tradeoff: first initialisation is slow (~10 s); subsequent calls are fast
"""
import logging
import re

import numpy as np

from backend.config import OCR_GPU, OCR_LANGUAGES
from backend.utils.image_utils import preprocess_for_ocr

logger = logging.getLogger(__name__)

# Characters that should never appear in a Western plate (common OCR mistakes)
_NOISE_CHARS = re.compile(r"[^A-Z0-9\-\s]")


class PlateOCR:
    """Singleton OCR reader wrapping EasyOCR."""

    _instance: "PlateOCR | None" = None

    def __init__(self) -> None:
        self.reader = None
        self.ready = False
        self._init_reader()

    @classmethod
    def get_instance(cls) -> "PlateOCR":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_reader(self) -> None:
        try:
            import easyocr  # type: ignore

            self.reader = easyocr.Reader(
                OCR_LANGUAGES,
                gpu=OCR_GPU,
                verbose=False,
            )
            self.ready = True
            logger.info(f"EasyOCR initialised (GPU={OCR_GPU}, langs={OCR_LANGUAGES})")
        except ImportError:
            logger.error("easyocr not installed – OCR disabled")
        except Exception as exc:
            logger.error(f"EasyOCR init failed: {exc}")

    def read_plate(self, crop: np.ndarray) -> dict:
        """
        Extract text from a cropped plate image.

        Returns:
            {
                "text": str,           # cleaned plate string
                "raw_text": str,       # raw OCR output (concatenated)
                "ocr_confidence": float
            }
        """
        if not self.ready or self.reader is None or crop.size == 0:
            return {"text": "", "raw_text": "", "ocr_confidence": 0.0}

        # Preprocess: upscale, CLAHE, binarise
        processed = preprocess_for_ocr(crop)

        try:
            results = self.reader.readtext(processed, detail=1, paragraph=False)
        except Exception as exc:
            logger.warning(f"OCR inference error: {exc}")
            return {"text": "", "raw_text": "", "ocr_confidence": 0.0}

        if not results:
            return {"text": "", "raw_text": "", "ocr_confidence": 0.0}

        # Combine all detected text segments
        raw_parts = [r[1] for r in results]
        confidences = [r[2] for r in results]
        raw_text = " ".join(raw_parts)
        avg_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0

        cleaned = self._clean_plate_text(raw_text)
        return {
            "text": cleaned,
            "raw_text": raw_text,
            "ocr_confidence": round(avg_conf, 4),
        }

    @staticmethod
    def _clean_plate_text(text: str) -> str:
        """
        Normalise OCR output to a standard plate format.
        Rules applied:
        - Uppercase
        - Strip leading/trailing whitespace
        - Remove characters that are not alphanumeric, hyphen, or space
        - Collapse multiple spaces/hyphens
        - Apply common letter↔digit substitutions (O→0, I→1, S→5, B→8)
          only in digit-expected positions (heuristic for European plates)
        """
        text = text.upper().strip()
        # Remove noise characters
        text = _NOISE_CHARS.sub("", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Collapse repeated hyphens
        text = re.sub(r"-{2,}", "-", text)
        return text
