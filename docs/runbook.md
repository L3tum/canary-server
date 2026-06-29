# Operational Runbook

This runbook covers day-to-day operations for the Canary ASR server, including scaling, model updates, monitoring, and incident response.

## Server Architecture Overview

- **ASR Server**: FastAPI app with global batch manager, per-GPU workers, rate limiting
- **Monitoring**: Prometheus metrics + Grafana dashboards
- **Deployment**: Docker Compose or Kubernetes

## Starting and Stopping the Server

### Start with Docker Compose (recommended)
```bash
docker compose up -d
```

### Start manually
```bash
INTERNAL_API_KEY="your-key" MODEL_NAME="nvidia/canary-1b-v2" \
python -m src.nemo_openai_server --host 0.0.0.0 --port 8000
```

### Health check
```bash
curl -H "Authorization: Bearer your-key" http://localhost:8000/health
```

Expected response includes `{"status":"ok","ready":true,"models_loaded":...}` plus GPU and queue details.

### Stop gracefully
```bash
# Using Docker Compose
docker compose down

# Or send SIGINT to the process
kill -INT <pid>
```

## Scaling

### Horizontal Scaling (More GPU Workers)

The server auto-scales within a single container by loading multiple model instances across available GPUs:

```bash
# Use all available GPUs (4 parallel instances)
PARALLEL_SIZE=4 docker compose up -d
```

Or with Docker run:
```bash
docker run -p 8000:8000 \
  -e PARALLEL_SIZE=4 \
  --gpus all \
  canary-asr-server
```

### Vertical Scaling (Larger Containers)

Increase `--max-batch-size` for larger batches (default 64):
```bash
python -m src.nemo_openai_server --max-batch-size 128
```

Increase `--thread-pool-size` for more CPU workers:
```bash
python -m src.nemo_openai_server --thread-pool-size 32
```

### Multi-Container Scaling

For high-traffic deployments, run multiple server instances behind a load balancer. Each container loads its own models independently.

## Model Updates

### Updating Model Weights

1. Stop the current server:
   ```bash
   docker compose down
   ```

2. Update the model in your storage (HuggingFace or local path):
   ```bash
   # If using HuggingFace, just pull the new version
   huggingface-cli download nvidia/canary-1b-v2 --local-dir /models/nvidia/canary-1b-v2
   ```

3. Restart the server:
   ```bash
   docker compose up -d
   ```

4. Verify with health check:
   ```bash
   curl -H "Authorization: Bearer your-key" http://localhost:8000/health
   ```

### Switching to a Different Model

1. Stop server
2. Download the new model:
   ```bash
   huggingface-cli download new-model-name --local-dir /models/new-model-name
   ```
3. Update `MODEL_NAME` environment variable:
   ```bash
   MODEL_NAME="new-model-name" docker compose up -d
   ```

## Monitoring

### Prometheus Metrics

Key metrics to monitor:

| Metric | Description | Alert Condition |
|--------|-------------|-----------------|
| `nemo_requests_total` | Total requests (by endpoint/status) | Sudden drop |
| `nemo_request_duration_seconds` | Request latency histogram | p99 > 10s |
| `nemo_queue_depth` | Pending request queue | > 100 |
| `nemo_batch_size` | Batch size distribution | Abnormally low |
| `nemo_model_loaded` | Model status (1 = loaded) | = 0 |

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (user: admin, password: admin).

Import the pre-configured ASR dashboard or create custom panels:
- **Request Throughput**: `rate(nemo_requests_total[5m])`
- **Latency P50/P90/P99**: `histogram_quantile(0.9, rate(nemo_request_duration_seconds_bucket[5m]))`
- **Queue Depth**: `nemo_queue_depth`
- **GPU Memory**: `nvidia_memory_used_bytes`

### Structured JSON Logs

All logs are in JSON format with `request_id` for tracing:
```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "level": "INFO",
  "logger": "src.nemo_openai_server",
  "message": "Processing batch of size 4 on GPU 0",
  "request_id": "req-abc123"
}
```

## Incident Response

### Server is unresponsive

1. Check readiness endpoint: `curl http://localhost:8000/healthz`
2. If GPU memory is full, kill the process and restart
3. Check logs for error patterns: `docker compose logs asr-server`
4. Restart: `docker compose restart asr-server`

### High Latency

1. Check queue depth: `curl http://localhost:8000/metrics | grep queue_depth`
2. If queue is growing, increase batch size or add more parallel instances
3. Check if CPU thread pool is saturated: increase `--thread-pool-size`
4. Verify GPU utilization: `nvidia-smi`

### Model Not Loading

1. Verify model exists in the expected path
2. Check GPU memory: `nvidia-smi`
3. Restart with `MODEL_TYPE` and `MODEL_TASK` environment variables
4. Check logs for model loading errors

### Rate Limiting Too Aggressive

Adjust rate limiting in the code or add rate limiting exceptions for trusted IPs.

## Troubleshooting

### "No models available" error

- Verify `MODEL_NAME` environment variable
- Check that model files are accessible in the container
- Check GPU memory availability

### "File too large" error

- Default limit: 500MB
- Adjust `max_file_size` in the transcription endpoint if needed

### Authentication failures

- Verify `INTERNAL_API_KEY` is set correctly
- Check that `Authorization: Bearer <key>` is included in requests

### Pre-commit hooks failing

```bash
# Update pre-commit hooks
pre-commit autoupdate

# Run hooks manually
pre-commit run --all-files
```

## Backup and Recovery

Model files should be backed up separately. Configuration (Docker Compose, environment variables) should be version-controlled.

## Performance Tuning

### Optimal Batch Size

Test with `benchmark.py`:
```bash
python tests/benchmark.py --url http://localhost:8000 --concurrent 20
```

Increase `--max-batch-size` and `--max-wait-ms` until latency increases.

### CPU Optimization

- Use dedicated thread pool: `--thread-pool-size` (default: CPU cores)
- Set `--use-fp16` for mixed precision (faster inference)
- Set `--use-torch-compile` for JIT compilation (first run slower, subsequent runs faster)