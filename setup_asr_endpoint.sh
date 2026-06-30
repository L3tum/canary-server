#!/bin/bash

# Setup script for NVIDIA Canary ASR endpoint

set -e  # Exit on any error

echo "Starting ASR endpoint setup..."

# Update system packages
echo "==> Updating system packages..."
sudo apt update

# Install required system tools
echo "==> Installing node exporter and basic tools..."
sudo apt -y install prometheus-node-exporter curl gnupg2 lsb-release
sudo systemctl enable prometheus-node-exporter || true

# Install Python dependencies with uv
echo "==> Setting up Python environment with uv..."
timer() { date +%s; }
elapsed() { echo "$(( $(timer) - $1 ))"; }
t0=$(timer)

# Create virtual environment and install dependencies
echo "Creating virtual environment with uv..."
uv venv
source .venv/bin/activate

# Install Python dependencies
echo "Installing Python packages with uv..."
uv pip install -U "nemo_toolkit[asr]" fastapi uvicorn prometheus-client python-multipart aiofiles soundfile

echo "Setup took $(elapsed $t0) seconds."

# Create server directory
echo "==> Creating server directory..."
mkdir -p "$HOME/nemo_openai_server"

# Create the main server module
echo "==> Writing server module..."
cat > "$HOME/nemo_openai_server/nemo_openai_server_module.py" <<'PYTHON_EOF'
#!/usr/bin/env python3
import os
import tempfile
import argparse
import time
import asyncio
import torch
from contextlib import asynccontextmanager
from typing import List, Dict, NamedTuple
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Request, Path
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter
import soundfile as sf

class QueuedRequest(NamedTuple):
    model: str
    tmpname: str
    source_lang: str
    target_lang: str
    future: asyncio.Future

class QueueManager:
    def __init__(self, models):
        self.queues = {}  # model -> asyncio.Queue
        self.processors = {}  # model -> asyncio.Task
        self.models = models  # Dictionary of model instances
    
    async def process_queue(self, model: str):
        queue = self.queues[model]
        model_instance = self.models[model]
        while True:
            try:
                req = await queue.get()
                if req is None:  # shutdown signal
                    break
                result = await self._process_request(req, model_instance)
                req.future.set_result(result)
            except Exception as e:
                if not req.future.done():
                    req.future.set_exception(e)
    
    async def _process_request(self, req: QueuedRequest, model_instance):
        audio_info = sf.info(req.tmpname)
        # return duration in seconds (audio_info.duration is in seconds)
        duration_seconds = audio_info.duration
        transcribe_kwargs = {"source_lang": req.source_lang, "target_lang": req.target_lang}
        outputs = model_instance.transcribe([req.tmpname], **transcribe_kwargs)
        text = outputs[0].text if outputs else ""
        return {"text": text, "model": req.model, "source_lang": req.source_lang,
                "target_lang": req.target_lang, "duration": duration_seconds}
    
    async def enqueue(self, model: str, tmpname: str, source_lang: str, target_lang: str):
        if model not in self.queues:
            self.queues[model] = asyncio.Queue()
            self.processors[model] = asyncio.create_task(self.process_queue(model))
        
        future = asyncio.Future()
        req = QueuedRequest(model, tmpname, source_lang, target_lang, future)
        await self.queues[model].put(req)
        return await future

# Lazy import NeMo model to avoid heavy import on module load if not needed
def load_model(model_name: str, device=None):
    from nemo.collections.asr.models import ASRModel
    model = ASRModel.from_pretrained(model_name=model_name)
    if device:
        model = model.to(device)
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--api-key", default=os.environ.get("INTERNAL_API_KEY"))
parser.add_argument("--model", default=os.environ.get("MODEL_NAME"))
parser.add_argument("--parallel-size", type=int, default=1, help="Number of parallel model instances")
args = parser.parse_args()

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

if not args.api_key:
    logger.warning(
        "No API key configured – all endpoints are unauthenticated. "
        "Set INTERNAL_API_KEY for production security."
    )

# metrics
REQUESTS = Counter('nemo_requests_total', 'Total number of requests', ['endpoint', 'status'])

# Lifespan context manager: load models before serving, clean up on shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize models on different GPUs
    app.state.models = {}
    app.state.model_names = []
    
    # Get available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found, loading model on CPU")
        # Load single model on CPU
        app.state.models[args.model] = load_model(args.model)
        app.state.model_names.append(args.model)
    else:
        # Distribute models across available GPUs
        for i in range(args.parallel_size):
            gpu_id = i % num_gpus
            device = torch.device(f"cuda:{gpu_id}")
            model_key = f"{args.model}_gpu{gpu_id}_instance{i}"
            print(f"Loading model instance {i} on {device}")
            app.state.models[model_key] = load_model(args.model, device)
            app.state.model_names.append(model_key)
    
    # Initialize queue manager with models
    app.state.queue_manager = QueueManager(app.state.models)
    
    try:
        yield
    finally:
        try:
            # attempt to free model references on shutdown
            del app.state.models
            del app.state.model_names
            del app.state.queue_manager
        except Exception:
            pass

app = FastAPI(title="NeMo OpenAI-compatible ASR", lifespan=lifespan)

def _check_auth(authorization: str):
    """Check authorization header. If no API key is configured, skip auth."""
    if args.api_key and authorization != f"Bearer {args.api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")

def _model_obj(name: str) -> Dict:
    return {
        "id": name,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "nemo",
        "permission": []
    }

def _available_models() -> List[str]:
    names = set()
    if args.model:
        names.add(args.model)
    loaded = getattr(app.state, "model_names", [])
    names.update(loaded)
    # return deterministic ordering
    return sorted([n for n in names if n])

@app.get("/health")
async def health(authorization: str = Header(None)):
    if args.api_key and authorization != f"Bearer {args.api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"status":"ok"}

@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/v1/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(None),
    source_lang: str = Form("es"),
    target_lang: str = Form("es"),
    authorization: str = Header(None),
):
    if args.api_key and authorization != f"Bearer {args.api_key}":
        REQUESTS.labels(endpoint='/v1/audio/transcriptions', status='401').inc()
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    model_name = model or args.model
    
    # Select a model instance (round-robin or default to first available)
    available_models = app.state.model_names
    if not available_models:
        raise HTTPException(status_code=500, detail="No models available")
    
    # If a specific model was requested, check if it exists
    if model and model in app.state.models:
        selected_model = model
    # If requesting the base model name, select one of the instances
    elif model == args.model or not model:
        # Simple round-robin selection based on queue sizes
        selected_model = min(available_models, key=lambda m: app.state.queue_manager.queues.get(m, asyncio.Queue()).qsize())
    else:
        raise HTTPException(status_code=400, detail=f"Model {model} not found")

    tmpfd, tmpname = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1] or ".wav")
    with os.fdopen(tmpfd, "wb") as f:
        f.write(await file.read())

    try:
        source_lang = source_lang or "es"
        target_lang = target_lang or "es"
        
        resp = await app.state.queue_manager.enqueue(selected_model, tmpname, source_lang, target_lang)
        REQUESTS.labels(endpoint='/v1/audio/transcriptions', status='200').inc()
        return JSONResponse(content=resp)
    except Exception as e:
        REQUESTS.labels(endpoint='/v1/audio/transcriptions', status='500').inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(tmpname)
        except:
            pass

# OpenAI-compatible model listing endpoints

@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    try:
        _check_auth(authorization)
    except HTTPException:
        REQUESTS.labels(endpoint='/v1/models', status='401').inc()
        raise
    models = [_model_obj(name) for name in _available_models()]
    resp = {"object": "list", "data": models}
    REQUESTS.labels(endpoint='/v1/models', status='200').inc()
    return JSONResponse(content=resp)
PYTHON_EOF

# Create the server wrapper
echo "==> Writing server wrapper..."
cat > "$HOME/nemo_openai_server/nemo_openai_server.py" <<'WRAPPER_EOF'
from nemo_openai_server_module import app
if __name__ == "__main__":
    import uvicorn, os
    host = os.environ.get("HOST","0.0.0.0")
    port = int(os.environ.get("PORT","8000"))
    uvicorn.run(app, host=host, port=port)
WRAPPER_EOF

# Make scripts executable
chmod +x "$HOME/nemo_openai_server/nemo_openai_server_module.py" "$HOME/nemo_openai_server/nemo_openai_server.py" || true
echo "Server files written to $HOME/nemo_openai_server and made executable."

# Create Prometheus configuration
echo "==> Creating Prometheus configuration..."
cat > /tmp/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'llm'
    static_configs:
      - targets: ['localhost:8000']
    metric_relabel_configs:
      - target_label: model_name
        replacement: "\${MODEL_NAME}"
      - target_label: model_type
        replacement: "\${MODEL_TYPE}"
      - target_label: model_task
        replacement: "\${MODEL_TASK}"
      - target_label: model_size
        replacement: "\${MODEL_SIZE}"
      - target_label: project
        replacement: "\${PROJECT}"
      - target_label: replica_id
        replacement: "\${SKYPILOT_SERVE_REPLICA_ID}"
      - target_label: service
        replacement: "stt"
      - target_label: provider
        replacement: "skypilot"
      - target_label: cloud
        replacement: "\${CLOUD}"
      - target_label: tech
        replacement: "\${TECH}"

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
    metric_relabel_configs:
      - target_label: model_name
        replacement: "\${MODEL_NAME}"
      - target_label: model_type
        replacement: "\${MODEL_TYPE}"
      - target_label: model_task
        replacement: "\${MODEL_TASK}"
      - target_label: model_size
        replacement: "\${MODEL_SIZE}"
      - target_label: project
        replacement: "\${PROJECT}"
      - target_label: replica_id
        replacement: "\${SKYPILOT_SERVE_REPLICA_ID}"
      - target_label: service
        replacement: "node_exporter"
      - target_label: provider
        replacement: "skypilot"
      - target_label: cloud
        replacement: "\${CLOUD}"
      - target_label: tech
        replacement: "\${TECH}"
EOF

sudo mkdir -p /etc/prometheus
sudo cp /tmp/prometheus.yml /etc/prometheus/prometheus.yml || true

echo "Setup complete! Run the ASR endpoint with: ./run_asr_endpoint.sh"