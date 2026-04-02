# PlateVision — Automatic License Plate Detection & Recognition

A complete **end-to-end** system for automatic license plate detection and OCR from images, videos, and live webcam/IP camera streams.

## Features

| Feature | Details |
|---|---|
| 🖼 **Image detection** | Upload JPG/PNG → annotated result + plate text in ~200 ms (CPU) |
| 🎬 **Video processing** | Upload any video → annotated output video + unique plate list |
| 📡 **Live stream** | MJPEG stream from local webcam or any RTSP/HTTP IP camera |
| 🧠 **Detection model** | YOLOv8s (fine-tuned on ~24k license-plate images) |
| 🔤 **OCR** | EasyOCR with CLAHE preprocessing (handles low light, blur, angle) |
| 💾 **History** | SQLite database storing every detection event |
| 🌐 **Web UI** | Dark-mode HTML/CSS/JS interface, no build step needed |
| 🔌 **REST API** | FastAPI with auto-generated Swagger docs at `/docs` |
| 🐳 **Docker** | `docker-compose up` for one-command deployment |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        Frontend                          │
│  index.html  upload-image  upload-video  live-camera     │
└─────────────────────┬────────────────────────────────────┘
                      │ REST API / MJPEG
┌─────────────────────▼────────────────────────────────────┐
│                     FastAPI Backend                       │
│  POST /api/image  POST /api/video  GET /api/stream/*      │
│  GET  /api/history  DELETE /api/history/{id}              │
└──────┬────────────────────┬──────────────────────────────┘
       │                    │
┌──────▼──────┐    ┌────────▼──────────┐    ┌─────────┐
│  YOLOv8s    │    │    EasyOCR        │    │ SQLite  │
│  Detector   │───▶│  (CLAHE + OCR)   │───▶│  DB     │
└─────────────┘    └───────────────────┘    └─────────┘
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/HoussamKhanfrii/Plates-detection.git
cd Plates-detection
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the API server

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The server pre-loads YOLOv8 + EasyOCR on startup (~20 s first run, subsequent fast).

### 3. Open the web interface

Open a browser and navigate to:
```
http://localhost:8000/ui
```
Or open `frontend/index.html` directly in a browser (set `API_BASE` in `app.js` if needed).

API docs: `http://localhost:8000/docs`

---

## Docker (recommended)

```bash
docker-compose up --build
```

The API is available at `http://localhost:8000`.
Custom weights are mounted from `./models/weights/best.pt`.

---

## Training a Custom Model

### 1. Download the dataset

```bash
python data/prepare_dataset.py --dest datasets/license_plates --api_key YOUR_ROBOFLOW_KEY
```

**Dataset:** [License Plate Recognition by Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e)
- ~24,000 annotated images in YOLOv8 format
- Diverse: day/night, angles, multiple countries
- Free for non-commercial use

### 2. Install training dependencies

```bash
pip install -r requirements-training.txt
```

### 3. Train

```bash
python training/train.py
# Or with overrides:
python training/train.py --model yolov8n.pt --epochs 50 --batch 8
```

Best weights are automatically saved to `models/weights/best.pt`.

### 4. Training config

Edit `training/config.yaml` to adjust hyperparameters, augmentation, and model size.

---

## Evaluation

```bash
python evaluation/evaluate.py --weights models/weights/best.pt --data datasets/license_plates/data.yaml
```

Outputs:
- **mAP@0.5**, **mAP@0.5:0.95**, **Precision**, **Recall**
- Latency benchmark (avg/P95 ms per image)
- JSON report saved to `evaluation/report.json`

---

## Standalone Inference (no server)

```bash
# Image
python inference/predict_image.py --source car.jpg --output result.jpg --show

# Video
python inference/predict_video.py --source traffic.mp4 --output result.mp4 --show

# Webcam
python inference/predict_realtime.py --source 0

# IP camera
python inference/predict_realtime.py --source rtsp://192.168.1.100:554/stream
```

---

## Project Structure

```
Plates-detection/
├── backend/                   # FastAPI application
│   ├── main.py                # App entry point, CORS, startup
│   ├── config.py              # Environment settings
│   ├── routers/               # API route handlers
│   │   ├── image.py           # POST /api/image
│   │   ├── video.py           # POST /api/video
│   │   ├── stream.py          # GET  /api/stream/*
│   │   └── history.py         # GET/DELETE /api/history
│   ├── services/              # Business logic
│   │   ├── detector.py        # YOLOv8 wrapper (singleton)
│   │   ├── ocr.py             # EasyOCR wrapper (singleton)
│   │   └── history.py         # DB CRUD operations
│   ├── models/schemas.py      # Pydantic request/response schemas
│   ├── database/db.py         # SQLAlchemy ORM + session management
│   └── utils/image_utils.py   # Preprocessing, annotation helpers
│
├── frontend/                  # Web interface (no build step)
│   ├── index.html             # Home / landing page
│   ├── upload-image.html      # Image detection page
│   ├── upload-video.html      # Video processing page
│   ├── live-camera.html       # Live MJPEG stream page
│   ├── history.html           # Detection history page
│   └── static/
│       ├── css/styles.css     # Global dark-mode styles
│       └── js/app.js          # Client-side logic
│
├── training/
│   ├── train.py               # YOLOv8 training script
│   └── config.yaml            # Hyperparameters & augmentation config
│
├── inference/
│   ├── predict_image.py       # Standalone image inference
│   ├── predict_video.py       # Standalone video inference
│   └── predict_realtime.py    # Standalone webcam/RTSP inference
│
├── evaluation/
│   └── evaluate.py            # mAP, OCR accuracy, latency benchmarks
│
├── data/
│   └── prepare_dataset.py     # Dataset download & validation
│
├── utils/
│   └── plate_utils.py         # Plate text normalisation helpers
│
├── models/weights/            # Place best.pt here after training
├── datasets/                  # Dataset downloaded here by prepare_dataset.py
├── uploads/                   # Runtime upload storage (git-ignored)
├── tests/
│   └── test_api.py            # pytest integration tests
├── docs/
│   ├── API.md                 # Full API reference
│   └── nginx.conf             # Production Nginx config
│
├── requirements.txt           # Core runtime dependencies
├── requirements-training.txt  # Additional training dependencies
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Configuration

All settings can be overridden with environment variables:

| Variable               | Default                        | Description                      |
|------------------------|--------------------------------|----------------------------------|
| `YOLO_WEIGHTS`         | `models/weights/best.pt`       | Path to trained weights          |
| `DETECTION_CONFIDENCE` | `0.4`                          | Minimum detection confidence     |
| `DETECTION_IOU`        | `0.45`                         | NMS IoU threshold                |
| `OCR_GPU`              | `false`                        | Enable GPU for EasyOCR           |
| `API_PORT`             | `8000`                         | Server port                      |
| `DATABASE_URL`         | `sqlite:///plates_history.db`  | SQLAlchemy DB URL                |
| `MAX_UPLOAD_SIZE_MB`   | `50`                           | Max upload size                  |
| `CORS_ORIGINS`         | `http://localhost:3000,...`    | Comma-separated allowed origins  |

---

## Running Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## Technical Choices

### Why YOLOv8?
- Best-in-class single-stage detector: ~50 ms/frame on CPU at 640px
- Excellent small object detection (license plates on large images)
- Ultralytics provides a clean Python API with built-in augmentation
- Easy to export to ONNX/TensorRT for production acceleration
- **Tradeoff:** YOLOv8n is 2x faster but 3-5 mAP points lower than YOLOv8s

### Why EasyOCR?
- Superior accuracy vs Tesseract on real-world plate images out of the box
- No system binary dependencies (unlike Tesseract)
- GPU-accelerated with CPU fallback
- Supports 80+ languages for international plates
- **Tradeoff:** Slower first init (~10 s); subsequent calls are fast (~50 ms)

### Why FastAPI?
- Async by design: handles concurrent uploads and streams efficiently
- Auto-generates OpenAPI/Swagger docs with zero extra work
- Pydantic validation on all inputs/outputs
- Simple to deploy with uvicorn or gunicorn

### Why SQLite?
- Zero-configuration; runs anywhere without a separate DB server
- Sufficient for local and small-production workloads
- Swap to PostgreSQL by changing `DATABASE_URL` (SQLAlchemy handles the rest)

---

## Handling Difficult Cases

| Challenge          | Mitigation                                                       |
|--------------------|------------------------------------------------------------------|
| Low light          | CLAHE preprocessing, HSV jitter augmentation during training     |
| Motion blur        | Blur augmentation in training config, temporal frame skip        |
| Angled plates      | Perspective + rotation augmentation, YOLO's anchor-free head     |
| Small plates       | Upscale small crops before OCR; train on multi-scale images      |
| Multiple vehicles  | YOLO returns all bounding boxes; all are processed               |
| Partial occlusion  | Mosaic + copy-paste augmentation helps robustness                |

---

## Future Improvements

- [ ] Export model to ONNX/TensorRT for 5-10x faster inference
- [ ] Add plate country/region classifier
- [ ] Integrate PaddleOCR as OCR fallback for Asian scripts
- [ ] WebSocket endpoint for lower-latency streaming
- [ ] Plate anonymisation / blurring mode for privacy compliance
- [ ] Rate limiting and API key authentication for production
- [ ] Multi-GPU training support
- [ ] React/Next.js frontend for richer UX

---

## License & Responsible Use

This system is intended for educational and research purposes.
License plate data is personally identifiable information (PII) in many jurisdictions.
Always comply with local privacy regulations (GDPR, CCPA, etc.) before deploying in production.
