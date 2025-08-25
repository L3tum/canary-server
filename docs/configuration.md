# Configuration Examples

## Basic Configuration

```bash
# Using environment variables
export INTERNAL_API_KEY=\"your-secret-api-key\"
export MODEL_NAME=\"nvidia/canary-1b-v2\"
python -m src.nemo_openai_server
```

## Parallel Processing Configuration

```bash
# Load 4 parallel instances across available GPUs
export INTERNAL_API_KEY=\"your-secret-api-key\"
export MODEL_NAME=\"nvidia/canary-1b-v2\"
export PARALLEL_SIZE=4
python -m src.nemo_openai_server
```

## Custom Host and Port

```bash
# Run on specific host and port
export INTERNAL_API_KEY=\"your-secret-api-key\"
python -m src.nemo_openai_server --host 127.0.0.1 --port 9000
```

## Complete Configuration Script

```bash
#!/bin/bash
# complete_config.sh

# Server configuration
export INTERNAL_API_KEY=\"your-secret-api-key\"
export MODEL_NAME=\"nvidia/canary-1b-v2\"
export MODEL_TYPE=\"audio\"
export MODEL_TASK=\"speech_to_text\"
export PARALLEL_SIZE=2
export HOST=\"0.0.0.0\"
export PORT=8000

# Run the server
python -m src.nemo_openai_server
```

## Docker Configuration

When running in Docker, you can pass environment variables:

```bash
docker run -e INTERNAL_API_KEY=\"your-secret-api-key\" \\
           -e MODEL_NAME=\"nvidia/canary-1b-v2\" \\
           -e PARALLEL_SIZE=4 \\
           -p 8000:8000 \\
           nemo-asr-server
```

## Using with docker-compose

```yaml
version: '3.8'
services:
  asr-server:
    image: nemo-asr-server
    environment:
      - INTERNAL_API_KEY=your-secret-api-key
      - MODEL_NAME=nvidia/canary-1b-v2
      - PARALLEL_SIZE=4
    ports:
      - \"8000:8000\"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```