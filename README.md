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
- [uv](https://docs.astral.sh/uv/) (recommended for environment management)
- CUDA-compatible GPU (optional, for GPU acceleration)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd nemo_openai_server
   ```

2. Create a virtual environment (recommended):
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

See [uv setup documentation](docs/uv_setup.md) for more detailed instructions on using uv with this project.

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