"""
Video inference script – standalone, no server required.

Usage:
    python inference/predict_video.py --source path/to/video.mp4
    python inference/predict_video.py --source path/to/video.mp4 --output result.mp4 --show
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
from backend.utils.image_utils import draw_detections

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def predict_video(
    source: str,
    output: str | None = None,
    show: bool = False,
    frame_skip: int = 3,
) -> dict:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {source}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    detector = PlateDetector.get_instance()
    ocr = PlateOCR.get_instance()

    unique_plates: set[str] = set()
    all_plates: list[dict] = []
    frame_idx = 0
    last_enriched: list[dict] = []

    t0 = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            raw_dets = detector.detect(frame)
            enriched: list[dict] = []
            for det in raw_dets:
                ocr_result = ocr.read_plate(det["crop"])
                entry = {
                    "bbox": det["bbox"],
                    "plate_text": ocr_result["text"],
                    "confidence": round(det["confidence"], 4),
                    "ocr_confidence": ocr_result["ocr_confidence"],
                }
                enriched.append(entry)
                all_plates.append(entry)
                if ocr_result["text"]:
                    unique_plates.add(ocr_result["text"])
            last_enriched = enriched

        annotated = draw_detections(frame, last_enriched)

        if writer:
            writer.write(annotated)
        if show:
            cv2.imshow("License Plate Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

        # Progress logging
        if frame_idx % 100 == 0:
            pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
            logger.info(f"  {frame_idx}/{total_frames} frames ({pct:.0f}%)  plates: {len(unique_plates)}")

    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = (time.perf_counter() - t0) * 1000

    stats = {
        "total_frames": total_frames,
        "processed_frames": frame_idx,
        "plates_detected": len(all_plates),
        "unique_plates": sorted(unique_plates),
        "elapsed_ms": round(elapsed, 1),
    }
    logger.info(f"\nDone in {elapsed:.0f} ms")
    logger.info(f"Unique plates found: {sorted(unique_plates)}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Detect license plates in a video")
    parser.add_argument("--source", required=True, help="Path to input video")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated video")
    parser.add_argument("--show", action="store_true", help="Display frames with OpenCV")
    parser.add_argument("--frame_skip", type=int, default=3, help="Process every Nth frame")
    args = parser.parse_args()

    predict_video(args.source, args.output, args.show, args.frame_skip)


if __name__ == "__main__":
    main()
