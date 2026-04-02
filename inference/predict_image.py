"""
Image inference script – standalone, no server required.

Usage:
    python inference/predict_image.py --source path/to/image.jpg
    python inference/predict_image.py --source path/to/image.jpg --output output.jpg --show
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.services.detector import PlateDetector
from backend.services.ocr import PlateOCR
from backend.utils.image_utils import draw_detections, read_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def predict_image(source: str, output: str | None = None, show: bool = False) -> list[dict]:
    image = read_image(source)
    detector = PlateDetector.get_instance()
    ocr = PlateOCR.get_instance()

    t0 = time.perf_counter()
    raw_dets = detector.detect(image)
    plates: list[dict] = []
    for det in raw_dets:
        ocr_result = ocr.read_plate(det["crop"])
        plates.append(
            {
                "bbox": det["bbox"],
                "plate_text": ocr_result["text"],
                "confidence": round(det["confidence"], 4),
                "ocr_confidence": ocr_result["ocr_confidence"],
            }
        )
    elapsed = (time.perf_counter() - t0) * 1000

    # ── Print results ─────────────────────────────────────────────────────────
    logger.info(f"Detected {len(plates)} plate(s) in {elapsed:.1f} ms")
    for i, p in enumerate(plates, 1):
        logger.info(
            f"  Plate {i}: '{p['plate_text']}'  "
            f"det_conf={p['confidence']:.3f}  "
            f"ocr_conf={p['ocr_confidence']:.3f}  "
            f"bbox={p['bbox']}"
        )

    # ── Annotate ──────────────────────────────────────────────────────────────
    annotated = draw_detections(image, plates)
    if output:
        cv2.imwrite(output, annotated)
        logger.info(f"Annotated image saved to: {output}")
    if show:
        cv2.imshow("License Plate Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return plates


def main():
    parser = argparse.ArgumentParser(description="Detect license plates in an image")
    parser.add_argument("--source", required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated image")
    parser.add_argument("--show", action="store_true", help="Display result with OpenCV")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence threshold")
    args = parser.parse_args()

    results = predict_image(args.source, args.output, args.show)
    if not results:
        logger.info("No plates detected.")


if __name__ == "__main__":
    main()
