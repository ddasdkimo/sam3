#!/bin/bash
# SAM3 Docker Startup Script

set -e

echo "=========================================="
echo "  SAM3 - Segment Anything with Concepts"
echo "=========================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found."
    echo "Creating from .env.example..."
    cp .env.example .env 2>/dev/null || true
    echo ""
    echo "Please edit .env and add your Hugging Face token:"
    echo "  1. Go to https://huggingface.co/settings/tokens"
    echo "  2. Create a new token"
    echo "  3. Request access to facebook/sam3 model"
    echo "  4. Add token to .env file"
    echo ""
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU may not be available."
    echo ""
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed."
    exit 1
fi

# Check Docker Compose
if ! docker compose version &> /dev/null; then
    echo "Error: Docker Compose is not installed."
    exit 1
fi

# Create directories
mkdir -p uploads outputs

echo "Starting SAM3 Docker container..."
echo ""

# Build and start
docker compose up --build "$@"
