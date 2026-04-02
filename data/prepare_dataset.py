"""
Dataset preparation script.

This script downloads and converts the license plate dataset from Roboflow
(or any compatible YOLO-format dataset) for training.

Dataset choice rationale:
    We use the "License Plate Recognition" dataset from Roboflow Universe:
    https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
    - ~24 000 annotated images with YOLO-format labels
    - Diverse lighting, angles, plate styles (US, EU, Asian)
    - Actively maintained; free for non-commercial use
    - Already split into train/valid/test (70/20/10)

    Secondary dataset (optional augmentation):
    OpenALPR benchmark dataset (CC BY 4.0) – 200k plates, useful for fine-tuning OCR

Usage:
    python data/prepare_dataset.py --dest datasets/license_plates
    python data/prepare_dataset.py --dest datasets/license_plates --api_key YOUR_KEY
"""
import argparse
import logging
import os
import shutil
import zipfile
from pathlib import Path

import requests
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ─── Roboflow public download URL (no-key version, 640px YOLOv8 export) ──────
ROBOFLOW_PROJECT = "license-plate-recognition-rxg4e"
ROBOFLOW_WORKSPACE = "roboflow-universe-projects"
ROBOFLOW_VERSION = 4


def download_roboflow(dest: Path, api_key: str | None) -> None:
    """
    Download the dataset using the Roboflow Python SDK if an API key is provided,
    otherwise print instructions for manual download.
    """
    try:
        from roboflow import Roboflow  # type: ignore

        if not api_key:
            raise ImportError("no key")
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        dataset = project.version(ROBOFLOW_VERSION).download("yolov8", location=str(dest))
        logger.info(f"Dataset downloaded to {dataset.location}")
    except Exception:
        logger.warning(
            "Roboflow SDK not available or no API key provided.\n"
            "To download automatically:\n"
            "  pip install roboflow\n"
            "  python data/prepare_dataset.py --dest datasets/license_plates --api_key YOUR_KEY\n\n"
            "Manual download:\n"
            "  1. Go to https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e\n"
            "  2. Export as 'YOLOv8' format\n"
            "  3. Extract to datasets/license_plates/\n"
        )


def verify_yolo_structure(dataset_dir: Path) -> bool:
    """Verify the dataset has the expected YOLOv8 directory structure."""
    required = ["train/images", "train/labels", "valid/images", "valid/labels"]
    missing = [r for r in required if not (dataset_dir / r).exists()]
    if missing:
        logger.warning(f"Missing directories: {missing}")
        return False
    return True


def generate_data_yaml(dataset_dir: Path) -> None:
    """
    Write a data.yaml for YOLOv8 training if one doesn't already exist.
    """
    yaml_path = dataset_dir / "data.yaml"
    if yaml_path.exists():
        logger.info(f"data.yaml already exists at {yaml_path}")
        return

    config = {
        "path": str(dataset_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["license_plate"],
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Generated {yaml_path}")


def count_dataset(dataset_dir: Path) -> dict:
    """Return image/label counts per split."""
    stats = {}
    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        n_images = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
        n_labels = len(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else 0
        stats[split] = {"images": n_images, "labels": n_labels}
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare license plate dataset")
    parser.add_argument(
        "--dest", type=str, default="datasets/license_plates",
        help="Destination directory for the dataset",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="Roboflow API key for automatic download",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preparing dataset at: {dest.resolve()}")
    download_roboflow(dest, args.api_key)

    if verify_yolo_structure(dest):
        generate_data_yaml(dest)
        stats = count_dataset(dest)
        for split, counts in stats.items():
            logger.info(f"  {split:6s}: {counts['images']} images, {counts['labels']} labels")
    else:
        logger.error("Dataset structure invalid. Please check the download.")


if __name__ == "__main__":
    main()
