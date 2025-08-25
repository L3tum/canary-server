#!/bin/bash

# Run script for NVIDIA Canary ASR endpoint

set -e  # Exit on any error

echo "Starting ASR endpoint..."

# Set default environment variables if not already set
export MODEL_NAME=${MODEL_NAME:-"nvidia/canary-1b-v2"}
export MODEL_TYPE=${MODEL_TYPE:-"audio"}
export MODEL_TASK=${MODEL_TASK:-"speech_to_text"}
export INTERNAL_API_KEY=${INTERNAL_API_KEY:-"your-api-key-here"}

echo "Using model: $MODEL_NAME"

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Start the server
echo "Starting NeMo OpenAI-compatible API server..."
echo "Server will be available at http://localhost:8000"
echo "Health check endpoint: http://localhost:8000/health"
echo "Metrics endpoint: http://localhost:8000/metrics"

# Check if parallel size is set in environment
PARALLEL_SIZE=${PARALLEL_SIZE:-1}

python "$HOME/nemo_openai_server/nemo_openai_server.py" \
  --api-key $INTERNAL_API_KEY \
  --model $MODEL_NAME \
  --parallel-size $PARALLEL_SIZE \
  --host 0.0.0.0 \
  --port 8000