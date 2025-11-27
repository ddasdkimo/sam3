# SAM3 Docker Image with Gradio Interface and CVAT Integration
# Base image: NVIDIA CUDA 13.0 for RTX 5090 (Blackwell) support
FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/root/.cache/torch
ENV HF_HOME=/root/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Install PyTorch 2.9.1 with CUDA 13.0 support (for RTX 5090 Blackwell sm_120)
RUN pip install --no-cache-dir torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 --break-system-packages && \
    pip install --no-cache-dir -e ".[notebooks]" --break-system-packages && \
    pip install --no-cache-dir \
    gradio>=4.0.0 \
    opencv-python-headless \
    av \
    psutil \
    fastapi \
    uvicorn[standard] \
    requests \
    pydantic \
    --break-system-packages

# Create directories for uploads and outputs
RUN mkdir -p /app/uploads /app/outputs

# Expose ports: 7860 for Gradio, 8000 for FastAPI
EXPOSE 7860 8000

# Set default command (can be overridden in docker-compose)
CMD ["python3", "app.py"]
