#!/bin/bash

# Test script for ASR endpoint

echo "Testing ASR endpoint health..."

# Check if the health endpoint responds
curl -v http://localhost:8000/health \
  # -H "Authorization: Bearer your-api-key"  # optional

echo ""
echo "If the above shows 'ok', the server is running correctly."
echo "To test with an actual audio file, use:"
echo "curl -X POST http://localhost:8000/v1/audio/transcriptions \\"
echo "  -H \"Authorization: Bearer your-api-key\" \\"  # optional
echo "  -F \"model=nvidia/canary-1b-v2\" \\"
echo "  -F \"file=@your_audio_file.wav\""