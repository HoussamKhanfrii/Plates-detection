FROM python:3.11-slim

# ─── System dependencies ──────────────────────────────────────────────────────
# libGL is needed by OpenCV; libglib2.0-0 by EasyOCR's torch internals
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ─── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ─── Python dependencies ──────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Application code ─────────────────────────────────────────────────────────
COPY . .

# Ensure upload/result directories exist inside the container
RUN mkdir -p uploads/images uploads/videos uploads/results models/weights

# ─── Environment defaults ────────────────────────────────────────────────────
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV OCR_GPU=false
ENV PYTHONUNBUFFERED=1

# ─── Expose & launch ──────────────────────────────────────────────────────────
EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
