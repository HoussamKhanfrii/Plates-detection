"""
Real-time webcam inference script – standalone, no server required.

Usage:
    python inference/predict_realtime.py                      # default webcam
    python inference/predict_realtime.py --source 1           # second webcam
    python inference/predict_realtime.py --source rtsp://...  # IP camera

Press 'q' to quit, 's' to save the current frame.
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


def run_realtime(source: int | str = 0, detect_every: int = 5) -> None:
    """
    Open a camera stream, run detection every `detect_every` frames,
    and display annotated results in a window.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open camera source: {source}")
        return

    detector = PlateDetector.get_instance()
    ocr = PlateOCR.get_instance()

    frame_idx = 0
    last_enriched: list[dict] = []
    fps_display = 0.0
    t_fps = time.perf_counter()

    logger.info("Press 'q' to quit, 's' to save the current frame.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Lost camera feed – attempting reconnect…")
            cap.release()
            time.sleep(0.5)
            cap = cv2.VideoCapture(source)
            continue

        # Run detection on every Nth frame
        if frame_idx % detect_every == 0:
            raw_dets = detector.detect(frame)
            enriched: list[dict] = []
            for det in raw_dets:
                ocr_result = ocr.read_plate(det["crop"])
                enriched.append(
                    {
                        "bbox": det["bbox"],
                        "plate_text": ocr_result["text"],
                        "confidence": round(det["confidence"], 4),
                        "ocr_confidence": ocr_result["ocr_confidence"],
                    }
                )
            last_enriched = enriched

        annotated = draw_detections(frame, last_enriched)

        # FPS overlay
        if frame_idx % 30 == 0:
            elapsed = time.perf_counter() - t_fps
            fps_display = 30 / elapsed if elapsed > 0 else 0
            t_fps = time.perf_counter()
        cv2.putText(
            annotated,
            f"FPS: {fps_display:.1f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 200, 255),
            2,
        )

        cv2.imshow("License Plate Detection – press Q to quit", annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            save_path = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(save_path, annotated)
            logger.info(f"Frame saved: {save_path}")

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-time license plate detection")
    parser.add_argument(
        "--source", default="0",
        help="Camera index (e.g. 0) or RTSP/HTTP URL",
    )
    parser.add_argument(
        "--detect_every", type=int, default=5,
        help="Run detection every N frames (trade speed for CPU usage)",
    )
    args = parser.parse_args()

    # Convert numeric source string to int
    source: int | str = args.source
    try:
        source = int(source)
    except ValueError:
        pass  # keep as string URL

    run_realtime(source, args.detect_every)


if __name__ == "__main__":
    main()
