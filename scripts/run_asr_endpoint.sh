#!/bin/bash

# Run script for NVIDIA Canary ASR endpoint

set -e  # Exit on any error

echo "Starting ASR endpoint..."

# Set default environment variables if not already set
export MODEL_NAME=${MODEL_NAME:-"nvidia/canary-1b-v2"}
export MODEL_TYPE=${MODEL_TYPE:-"audio"}
export MODEL_TASK=${MODEL_TASK:-"speech_to_text"}
# API key is optional (omit for no authentication)
export INTERNAL_API_KEY=${INTERNAL_API_KEY:-}

echo "Using model: $MODEL_NAME"

# Check if parallel size is set in environment
PARALLEL_SIZE=${PARALLEL_SIZE:-1}

python -m src.nemo_openai_server \
  --api-key $INTERNAL_API_KEY \
  --model $MODEL_NAME \
  --parallel-size $PARALLEL_SIZE \
  --host 0.0.0.0 \
  --port 8000