# NVIDIA Canary ASR Endpoint

OpenAI-compatible ASR (Automatic Speech Recognition) server powered by NVIDIA's Canary 1B v2 model through NeMo toolkit. Production-ready with global batch processing, GPU parallelism, JSON logging, and comprehensive monitoring.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/.../canary-server.git
cd canary-server
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Run the server
INTERNAL_API_KEY="your-secret-key" MODEL_NAME="nvidia/canary-1b-v2" python -m src.nemo_openai_server
```

## API Documentation

The server provides an OpenAI-compatible API with the following endpoints:

### Authentication

All endpoints except `/metrics` and `/healthz` require a Bearer token:

```
Authorization: Bearer YOUR_API_KEY
```

Set the `INTERNAL_API_KEY` environment variable when starting the server.

### Endpoints

#### POST `/v1/audio/transcriptions`

Transcribe audio files to text.

**Request:**
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your-api-key" \
  -F "model=nvidia/canary-1b-v2" \
  -F "file=@audio.wav" \
  -F "source_lang=en" \
  -F "target_lang=en"
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `model` | string | No | Model ID (default: server's model) |
| `file` | file | Yes | Audio file (wav, mp3, etc.) |
| `source_lang` | string | No | Source language code (default: "en") |
| `target_lang` | string | No | Target language code (default: "en") |

**Response (200 OK):**
```json
{
  "text": "Transcribed text here",
  "model": "nvidia/canary-1b-v2",
  "source_lang": "en",
  "target_lang": "en",
  "duration": 5.2
}
```

**Error Codes:**
| Code | Error | Description |
|------|-------|-------------|
| 400 | Bad Request | Invalid audio file, unsupported format, invalid language |
| 401 | Unauthorized | Missing or invalid API key |
| 413 | Payload Too Large | File exceeds 500MB limit |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | No models loaded or processing error |

**Rate Limiting:**
Default: 30 requests per minute per IP address. Adjust with `--rate-limit` or `RATE_LIMIT` (for example, `100/minute`).

#### GET `/v1/models`

List available models.

```bash
curl -H "Authorization: Bearer your-api-key" http://localhost:8000/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "nvidia/canary-1b-v2",
      "object": "model",
      "created": 1690000000,
      "owned_by": "nvidia"
    }
  ]
}
```

#### GET `/health`

Health check with GPU status and memory usage.

```bash
curl -H "Authorization: Bearer your-api-key" http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "request_id": "req-abc123",
  "ready": true,
  "models_loaded": 1,
  "gpu_count": 1,
  "gpu_info": {},
  "queue_depth": 0,
  "cuda_available": true
}
```

#### GET `/healthz`

Readiness check for Docker and load balancers (no auth required). Returns `200 OK` when the server is ready and `503` while models are loading.

#### GET `/metrics`

Prometheus metrics (no auth required). Returns metrics including:
- `nemo_requests_total` — total request count
- `nemo_request_duration_seconds` — request latency histogram
- `nemo_queue_depth` — current queue depth
- `nemo_batch_size` — batch size distribution
- `nemo_model_loaded` — model status

### OpenAPI / Swagger UI

Interactive API documentation available at:
```
http://localhost:8000/docs
```

Or the raw OpenAPI schema at:
```
http://localhost:8000/openapi.json
```

## Architecture

The server uses an **in-memory global batch manager** with per-GPU workers for maximum throughput:

1. **Request ingestion**: Incoming requests are queued in memory (no temp files).
2. **Batch formation**: GlobalBatchManager collects requests and forms optimal batches.
3. **GPU distribution**: Batches are distributed round-robin across GPU workers.
4. **Parallel processing**: Multiple requests processed simultaneously on different GPUs.
5. **Result aggregation**: Results returned to original requesters.

### Key Features

- **No temp files**: Audio data processed entirely in memory
- **Global batching**: Efficient batch sizes regardless of GPU assignment
- **Per-GPU workers**: Dedicated async workers per GPU
- **FP16 inference**: Optional mixed-precision for faster inference
- **torch.compile**: Optional JIT compilation for performance
- **Structured JSON logging**: Each request gets a unique trace ID
- **Rate limiting**: Protects against abuse via slowapi
- **Input validation**: Audio format, file size (500MB limit), language codes

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `INTERNAL_API_KEY` | API key (required) | `your-api-key-here` |
| `MODEL_NAME` | Model identifier | `nvidia/canary-1b-v2` |
| `MODEL_TYPE` | Model type | `audio` |
| `MODEL_TASK` | Model task | `speech_to_text` |
| `PARALLEL_SIZE` | Parallel model instances | `1` |
| `HOST` | Server bind address | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `RATE_LIMIT` | Per-IP transcription rate limit | `30/minute` |
| `MAX_BATCH_SIZE` | Maximum transcription batch size | `64` |
| `MAX_WAIT_MS` | Maximum wait time for batching | `30` |
| `THREAD_POOL_SIZE` | CPU worker thread pool size | CPU cores |
| `USE_FP16` | Enable FP16 when set to `true` | `false` |
| `USE_TORCH_COMPILE` | Enable `torch.compile` when set to `true` | `false` |

### Command-Line Arguments

| Flag | Description |
|------|-------------|
| `--host` | Server host (0.0.0.0) |
| `--port` | Server port (8000) |
| `--api-key` | API key (overrides env) |
| `--model` | Model name (overrides env) |
| `--parallel-size` | Number of model instances |
| `--max-batch-size` | Max batch size (64) |
| `--max-wait-ms` | Max wait time for batching (30ms) |
| `--thread-pool-size` | CPU thread pool (auto) |
| `--rate-limit` | Per-IP transcription rate limit (30/minute) |
| `--use-fp16` | Use FP16 precision |
| `--use-torch-compile` | Enable torch.compile |

### Supported Languages

The server supports over 40 languages. Full list:
`en, es, fr, de, it, pt, zh, ja, ko, ru, ar, hi, th, vi, tr, pl, nl, sv, fi, da, no, el, he, cs, sk, hu, ro, bg, hr, sr, sl, et, lv, lt, uk, fa, id, ms, sw, tl`

Plus `auto` for automatic language detection (source_lang only).

## Docker Deployment

### Docker Compose (with monitoring)
```bash
# Build and run with Prometheus + Grafana
docker compose up --build

# Services:
# - ASR server (port 8000)
# - Prometheus (port 9090)
# - Grafana (port 3000)
```

See [Dockerfile](Dockerfile) for the container build.

### Custom Docker Image
```bash
docker build -t canary-asr-server .
docker run -p 8000:8000 \
  -e INTERNAL_API_KEY="your-key" \
  -e MODEL_NAME="nvidia/canary-1b-v2" \
  -e PARALLEL_SIZE=2 \
  --gpus all \
  canary-asr-server
```

## Testing & Quality

### Run Tests
```bash
make test
```

### Linting
```bash
make lint
```

### Pre-commit Hooks
```bash
pip install pre-commit
pre-commit install
```

### Load Testing
```bash
python tests/benchmark.py --host localhost --port 8000 --files path/to/audio.wav
```

## Documentation

- [Parallel Processing Guide](docs/parallel_processing.md)
- [CPU Optimization Guide](docs/cpu_optimization.md)
- [Operational Runbook](docs/runbook.md) — scaling, monitoring, model updates
- [uv Setup Guide](docs/uv_setup.md)

## License

Apache License 2.0