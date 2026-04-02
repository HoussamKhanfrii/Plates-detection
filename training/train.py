"""
Training script for the YOLOv8 license plate detector.

Usage:
    # Train with defaults from training/config.yaml
    python training/train.py

    # Override specific settings
    python training/train.py --epochs 50 --batch 8 --model yolov8n.pt

    # Resume from last checkpoint
    python training/train.py --resume

After training, the best weights are automatically copied to:
    models/weights/best.pt
"""
import argparse
import logging
import shutil
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "training" / "config.yaml"
WEIGHTS_OUT = ROOT / "models" / "weights" / "best.pt"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def train(overrides: dict) -> None:
    """
    Run YOLOv8 training.

    Design notes:
    - We use YOLOv8s as the default: a good tradeoff between mAP and inference speed.
      yolov8n is ~2× faster but ~3 mAP points lower on COCO.
    - Aggressive augmentation (blur, HSV jitter, mosaic) is critical for real-world
      robustness on low-light, motion-blurred, or angled plates.
    - EarlyStopping (patience=20) prevents overfitting on small datasets.
    - Best weights are saved automatically by Ultralytics at runs/train/*/weights/best.pt
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        logger.error(
            "ultralytics not installed. Install with:\n"
            "  pip install ultralytics"
        )
        return

    cfg = load_config()
    cfg.update(overrides)          # CLI flags override config.yaml

    # Resolve relative data path to absolute
    data_path = Path(cfg.get("data", "datasets/license_plates/data.yaml"))
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    cfg["data"] = str(data_path)

    if not data_path.exists():
        logger.error(
            f"data.yaml not found at {data_path}.\n"
            "Run: python data/prepare_dataset.py --dest datasets/license_plates"
        )
        return

    model_name = cfg.pop("model", "yolov8s.pt")
    logger.info(f"Loading base model: {model_name}")
    model = YOLO(model_name)

    logger.info("Starting training with config:")
    for k, v in cfg.items():
        logger.info(f"  {k}: {v}")

    results = model.train(**cfg)

    # ── Copy best weights to the canonical location ───────────────────────────
    run_dir = Path(results.save_dir)
    best_src = run_dir / "weights" / "best.pt"
    if best_src.exists():
        WEIGHTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_src, WEIGHTS_OUT)
        logger.info(f"Best weights saved to: {WEIGHTS_OUT}")
    else:
        logger.warning("best.pt not found in run directory.")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 license plate detector")
    parser.add_argument("--model", type=str, help="YOLO model variant, e.g. yolov8s.pt")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--device", type=str, help="Device: '' | 'cpu' | '0' | '0,1'")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument(
        "--data", type=str, help="Path to dataset data.yaml"
    )
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None and not isinstance(v, bool)}
    if args.resume:
        overrides["resume"] = True

    train(overrides)


if __name__ == "__main__":
    main()
