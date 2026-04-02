"""
Evaluation script.

Computes:
  Detection:  mAP@0.5, mAP@0.5:0.95, precision, recall
  OCR:        character accuracy, word accuracy (plate-level exact match)
  Latency:    average inference time per image (ms)

Usage:
    python evaluation/evaluate.py --weights models/weights/best.pt --data datasets/license_plates/data.yaml
    python evaluation/evaluate.py --weights models/weights/best.pt --images path/to/test/images --labels path/to/test/labels
"""
import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent


# ─── Detection evaluation ─────────────────────────────────────────────────────
def run_yolo_validation(weights: str, data_yaml: str, imgsz: int = 640) -> dict:
    """Run ultralytics built-in val pipeline and return metrics dict."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        logger.error("ultralytics not installed")
        return {}

    model = YOLO(weights)
    results = model.val(data=data_yaml, imgsz=imgsz, verbose=True)
    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }
    return metrics


# ─── OCR evaluation ───────────────────────────────────────────────────────────
def char_accuracy(pred: str, gt: str) -> float:
    """
    Normalised edit-distance-based character accuracy.
    acc = 1 - (edit_distance / max_len)
    """
    import difflib
    if not gt:
        return 1.0 if not pred else 0.0
    ratio = difflib.SequenceMatcher(None, pred.upper(), gt.upper()).ratio()
    return round(ratio, 4)


def evaluate_ocr(
    images_dir: Path,
    labels_dir: Path,
    ocr_reader,
) -> dict:
    """
    Evaluate OCR on ground-truth cropped plate images.
    Expects a text file per image with the plate string as content.
    """
    from backend.utils.image_utils import preprocess_for_ocr

    image_paths = sorted(images_dir.glob("*"))
    n_total = 0
    n_exact = 0
    char_accs: list[float] = []
    latencies: list[float] = []

    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        gt_text = label_path.read_text().strip().upper()

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        processed = preprocess_for_ocr(img)

        t0 = time.perf_counter()
        result = ocr_reader.readtext(processed, detail=1)
        latencies.append((time.perf_counter() - t0) * 1000)

        pred_text = " ".join([r[1] for r in result]).strip().upper()

        n_total += 1
        if pred_text == gt_text:
            n_exact += 1
        char_accs.append(char_accuracy(pred_text, gt_text))

    if n_total == 0:
        return {"ocr_samples": 0, "note": "No OCR test samples found"}

    return {
        "ocr_samples": n_total,
        "exact_match_rate": round(n_exact / n_total, 4),
        "avg_char_accuracy": round(float(np.mean(char_accs)), 4),
        "avg_ocr_latency_ms": round(float(np.mean(latencies)), 2),
    }


# ─── Latency benchmark ────────────────────────────────────────────────────────
def benchmark_latency(weights: str, n_runs: int = 100, imgsz: int = 640) -> dict:
    """Measure average detection inference time on a synthetic image."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        return {}

    model = YOLO(weights)
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        model.predict(dummy, verbose=False)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "runs": n_runs,
        "avg_latency_ms": round(float(np.mean(times)), 2),
        "p95_latency_ms": round(float(np.percentile(times, 95)), 2),
        "min_latency_ms": round(float(np.min(times)), 2),
        "max_latency_ms": round(float(np.max(times)), 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate license plate detection system")
    parser.add_argument(
        "--weights", type=str, default="models/weights/best.pt",
        help="Path to YOLOv8 weights",
    )
    parser.add_argument(
        "--data", type=str, default="datasets/license_plates/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size"
    )
    parser.add_argument(
        "--latency_runs", type=int, default=100,
        help="Number of runs for latency benchmark",
    )
    args = parser.parse_args()

    weights = str(ROOT / args.weights) if not Path(args.weights).is_absolute() else args.weights
    data_yaml = str(ROOT / args.data) if not Path(args.data).is_absolute() else args.data

    report = {}

    # ── Detection metrics ─────────────────────────────────────────────────────
    if Path(data_yaml).exists():
        logger.info("Running YOLOv8 validation…")
        det_metrics = run_yolo_validation(weights, data_yaml, args.imgsz)
        report["detection"] = det_metrics
        if det_metrics:
            logger.info(
                f"mAP@0.5={det_metrics['mAP50']:.3f}  "
                f"mAP@0.5:0.95={det_metrics['mAP50_95']:.3f}  "
                f"P={det_metrics['precision']:.3f}  "
                f"R={det_metrics['recall']:.3f}"
            )
    else:
        logger.warning(f"data.yaml not found: {data_yaml}")

    # ── Latency benchmark ─────────────────────────────────────────────────────
    if Path(weights).exists():
        logger.info(f"Running latency benchmark ({args.latency_runs} runs)…")
        lat = benchmark_latency(weights, n_runs=args.latency_runs, imgsz=args.imgsz)
        report["latency"] = lat
        logger.info(
            f"Avg latency: {lat.get('avg_latency_ms', 'N/A')} ms  "
            f"P95: {lat.get('p95_latency_ms', 'N/A')} ms"
        )

    # ── Print full report ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))

    # Save report
    report_path = ROOT / "evaluation" / "report.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
