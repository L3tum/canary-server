# NVIDIA Canary ASR Endpoint

This repository provides an OpenAI-compatible ASR (Automatic Speech Recognition) endpoint using NVIDIA's Canary 1B v2 model through the NeMo toolkit.

## Features

- OpenAI-compatible API endpoints
- Support for NVIDIA's Canary ASR models
- GPU-parallel inference support
- Prometheus metrics collection
- Health check endpoint

## Prerequisites

- Python 3.8 or higher
- Conda (recommended for environment management)
- CUDA-compatible GPU (optional, for GPU acceleration)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd nemo_openai_server
   ```

2. Create a virtual environment (recommended):
   ```bash
   conda create -n nemo-asr python=3.10
   conda activate nemo-asr
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

The server can be configured using environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `INTERNAL_API_KEY` | API key for authentication (required) | `your-api-key-here` |
| `MODEL_NAME` | The model to use | `nvidia/canary-1b-v2` |
| `MODEL_TYPE` | Model type | `audio` |
| `MODEL_TASK` | Model task | `speech_to_text` |
| `PARALLEL_SIZE` | Number of parallel model instances | `1` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |

### Parallel Processing

The server supports parallel processing by loading multiple instances of the same model across available GPUs. This feature improves throughput by allowing simultaneous transcription requests to be processed on different GPU devices.

To enable parallel processing, set the `PARALLEL_SIZE` environment variable or use the `--parallel-size` command line argument:

```bash
export PARALLEL_SIZE=4
python -m src.nemo_openai_server
```

or

```bash
python -m src.nemo_openai_server --parallel-size 4
```

See [Parallel Processing Documentation](docs/parallel_processing.md) for more details.

## Running the Server

Start the server with:
```bash
python -m src.nemo_openai_server --api-key your-secret-api-key
```

Or with environment variables:
```bash
export INTERNAL_API_KEY="your-secret-api-key"
export MODEL_NAME="nvidia/canary-1b-v2"
export PARALLEL_SIZE=2
python -m src.nemo_openai_server
```

The server will be available at `http://localhost:8000` with the following endpoints:
- `/health` - Health check endpoint
- `/metrics` - Prometheus metrics
- `/v1/audio/transcriptions` - ASR transcription endpoint (OpenAI-compatible)
- `/v1/models` - Model listing endpoint (OpenAI-compatible)

## Usage

To transcribe an audio file:
```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer your-api-key-here" \
  -F "model=nvidia/canary-1b-v2" \
  -F "file=@your_audio_file.wav"
```

## Testing

Run the health check:
```bash
curl -v http://localhost:8000/health \
  -H "Authorization: Bearer your-api-key-here"
```

## Metrics

Prometheus metrics are available at `/metrics` endpoint. To enable system metrics collection, install and run Prometheus node exporter:
```bash
sudo apt install prometheus-node-exporter
sudo systemctl enable prometheus-node-exporter
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Batching and Throughput Optimization

The server supports request batching to improve GPU utilization and overall throughput when processing multiple audio requests concurrently. This feature batches multiple queued requests per model instance into a single NeMo transcribe() call when possible.

### Batching Configuration

Batching can be configured using the following command line arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--max-batch-size` | Maximum number of audio requests per batch per model instance | `4` |
| `--max-batch-delay-ms` | Maximum waiting time after first item (in milliseconds) to accumulate a batch | `25` |
| `--disable-batching` | Disable batching and fall back to per-request behavior | `false` |

### Example Usage

Enable batching with custom settings:
```bash
python -m src.nemo_openai_server \
  --api-key your-secret-api-key \
  --parallel-size 2 \
  --max-batch-size 8 \
  --max-batch-delay-ms 50
```

Disable batching for latency-sensitive applications:
```bash
python -m src.nemo_openai_server \
  --api-key your-secret-api-key \
  --disable-batching
```

### Batching Behavior

- **Language Matching**: Only requests with matching `source_lang` and `target_lang` are batched together
- **Time Limits**: Batches are processed when either `max_batch_size` is reached or `max_batch_delay_ms` has elapsed since the first request
- **Fairness**: Requests with different language pairs are re-queued fairly and not skipped
- **Response Format**: Individual responses include additional `batch_size` and `batch_position` fields

### Performance Trade-offs

**Throughput vs Latency:**
- **Higher batch sizes** (8-16): Better GPU utilization and throughput, but higher latency for individual requests
- **Lower batch sizes** (2-4): Better latency with moderate throughput improvement
- **Shorter delays** (10-25ms): Lower latency impact when queue is not full
- **Longer delays** (50-100ms): Better batching efficiency but higher minimum latency

### Tuning Recommendations

1. **High-throughput scenarios**: Use `--max-batch-size 8` or higher with `--max-batch-delay-ms 50-100`
2. **Latency-sensitive applications**: Use `--max-batch-size 2-4` with `--max-batch-delay-ms 10-25`
3. **Mixed workloads**: Start with defaults (`--max-batch-size 4 --max-batch-delay-ms 25`) and adjust based on monitoring
4. **Real-time applications**: Consider `--disable-batching` if latency is more critical than throughput

### Monitoring

Batch information is logged at INFO level:
```
Processing batch: model=nvidia/canary-1b-v2_gpu0_instance0, batch_size=4, queue_remaining=2, source_lang=en, target_lang=en
Batch completed: model=nvidia/canary-1b-v2_gpu0_instance0, batch_size=4, latency=0.245s
```

Use these logs to tune batching parameters based on your workload characteristics.