# ==============================================================================
# Production Dockerfile for VisionAudit AI Microservice
# Multi-stage optimized build with Python 3.11-slim, headless OpenCV, & YOLOv8
# ==============================================================================

FROM python:3.11-slim

# Set environment variables for Python runtime and Ultralytics
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_CONFIG_DIR=/app/.ultralytics

# Install system dependencies required for OpenCV headless and PyTorch/YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Create unprivileged application user
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/bash -m appuser

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n weights into the container image to prevent cold-start download at runtime
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copy application source code
COPY . .

# Adjust permissions for non-root user (including Ultralytics cache directory)
RUN mkdir -p /app/.ultralytics && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose microservice port
EXPOSE 8000

# Docker healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI application via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
