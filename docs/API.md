# API Reference

## Base URL
```
http://localhost:8000
```

Interactive Swagger UI: `http://localhost:8000/docs`
ReDoc:                  `http://localhost:8000/redoc`

---

## Endpoints

### `GET /health`
Health check – returns model and OCR status.

**Response 200**
```json
{
  "status": "ok",
  "model_loaded": true,
  "ocr_ready": true,
  "version": "1.0.0"
}
```

---

### `POST /api/image`
Detect license plates in an uploaded image.

**Request** – `multipart/form-data`
| Field | Type   | Required | Description          |
|-------|--------|----------|----------------------|
| file  | binary | ✓        | JPG/PNG/BMP/WebP image |

**Response 200**
```json
{
  "source_type": "image",
  "filename": "car.jpg",
  "plates": [
    {
      "plate_text": "AB123CD",
      "confidence": 0.934,
      "ocr_confidence": 0.871,
      "bbox": [120, 340, 310, 390]
    }
  ],
  "plates_count": 1,
  "inference_time_ms": 182.4,
  "result_image_url": "/results/result_<uid>.jpg"
}
```

**Error responses**
| Code | Reason                |
|------|-----------------------|
| 415  | Unsupported file type |
| 422  | Cannot decode image   |

---

### `POST /api/video`
Process an uploaded video file frame by frame.

**Request** – `multipart/form-data`
| Field | Type   | Required | Description          |
|-------|--------|----------|----------------------|
| file  | binary | ✓        | MP4/AVI/MOV/MKV/WebM |

**Response 200**
```json
{
  "source_type": "video",
  "filename": "traffic.mp4",
  "plates_detected": 42,
  "unique_plates": ["AB123", "XY789"],
  "total_frames": 1200,
  "processed_frames": 400,
  "inference_time_ms": 38421.0,
  "result_video_url": "/results/result_<uid>.mp4"
}
```

---

### `GET /api/stream/webcam`
Stream annotated MJPEG video from the local webcam.

**Query params**
| Param          | Type | Default | Description          |
|----------------|------|---------|----------------------|
| camera_index   | int  | 0       | OpenCV camera index  |

**Response** – `multipart/x-mixed-replace; boundary=frame`  
Consume with an `<img src="...">` tag or any MJPEG client.

---

### `GET /api/stream/ip`
Stream annotated MJPEG video from an IP / RTSP camera.

**Query params**
| Param | Type   | Required | Description                |
|-------|--------|----------|----------------------------|
| url   | string | ✓        | RTSP or HTTP camera URL    |

---

### `GET /api/history`
List past detection records (newest first).

**Query params**
| Param | Type | Default | Description          |
|-------|------|---------|----------------------|
| skip  | int  | 0       | Pagination offset    |
| limit | int  | 50      | Max records returned |

**Response 200** – Array of `DetectionHistoryItem`
```json
[
  {
    "id": 7,
    "source_type": "image",
    "filename": "car.jpg",
    "plate_text": "AB123CD",
    "confidence": 0.934,
    "ocr_confidence": 0.871,
    "result_path": "/app/uploads/results/result_abc.jpg",
    "plates_count": 1,
    "created_at": "2024-06-01T12:34:56+00:00"
  }
]
```

---

### `DELETE /api/history/{record_id}`
Delete a single detection record.

**Response** `204 No Content`  
**Error** `404` if record not found.

---

## Static files

| Path                          | Description                        |
|-------------------------------|------------------------------------|
| `/results/<filename>`         | Download annotated result images/videos |
| `/ui/`                        | Serve the built-in HTML frontend   |
| `/docs`                       | Swagger UI                         |
| `/redoc`                      | ReDoc documentation                |
