"""
Utility functions for image preprocessing and postprocessing.
"""
import cv2
import numpy as np
from pathlib import Path


def read_image(path: str | Path) -> np.ndarray:
    """Read an image from disk; raise ValueError if it can't be decoded."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image: {path}")
    return img


def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """
    Enhance a cropped plate image for better OCR accuracy.
    Pipeline:
        1. Upscale small crops (helps Tesseract/EasyOCR on tiny plates)
        2. Convert to grayscale
        3. Apply CLAHE for contrast normalisation (handles low-light)
        4. Gentle Gaussian blur to reduce JPEG artefacts
        5. Adaptive thresholding for binarisation
    """
    h, w = crop.shape[:2]

    # 1. Upscale if the plate is very small
    min_width = 120
    if w < min_width:
        scale = min_width / w
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE – contrast limited adaptive histogram equalisation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    # 4. Light Gaussian blur to suppress noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 5. Adaptive thresholding preserves details under varying illumination
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2,
    )

    return binary


def draw_detections(
    image: np.ndarray,
    plates: list[dict],
    box_color: tuple = (0, 200, 0),
    text_color: tuple = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes and plate text on an image.

    plates: list of dicts with keys: bbox, plate_text, confidence
    """
    annotated = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6

    for plate in plates:
        x1, y1, x2, y2 = plate["bbox"]
        text = plate.get("plate_text", "")
        conf = plate.get("confidence", 0.0)
        label = f"{text}  {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        # Draw filled label background for readability
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        bg_y1 = max(y1 - text_h - baseline - 4, 0)
        cv2.rectangle(annotated, (x1, bg_y1), (x1 + text_w + 4, y1), box_color, cv2.FILLED)
        cv2.putText(
            annotated, label,
            (x1 + 2, y1 - baseline - 2),
            font, font_scale, text_color, thickness, cv2.LINE_AA,
        )

    return annotated


def encode_image_to_bytes(image: np.ndarray, ext: str = ".jpg") -> bytes:
    """Encode a NumPy image array to JPEG/PNG bytes."""
    success, buffer = cv2.imencode(ext, image)
    if not success:
        raise RuntimeError("Failed to encode image")
    return buffer.tobytes()


def safe_crop(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Crop with clamped coordinates to avoid out-of-bounds slicing."""
    h, w = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return image[y1:y2, x1:x2]
