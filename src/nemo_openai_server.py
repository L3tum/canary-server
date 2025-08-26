#!/usr/bin/env python3
"""
NeMo OpenAI-compatible ASR Server

This module provides an OpenAI-compatible ASR (Automatic Speech Recognition) endpoint 
using NVIDIA's Canary 1B v2 model through the NeMo toolkit.
"""

import os
import tempfile
import argparse
import time
import asyncio
import torch
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, NamedTuple
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Request, Path
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter
import soundfile as sf

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class QueuedRequest(NamedTuple):
    model: str
    tmpname: str
    source_lang: str
    target_lang: str
    future: asyncio.Future

class QueueManager:
    def __init__(self, models, use_round_robin=False):
        self.queues = {}  # model -> asyncio.Queue
        self.processors = {}  # model -> asyncio.Task
        self.models = models  # Dictionary of model instances
        self.use_round_robin = use_round_robin
        self.round_robin_lock = asyncio.Lock()
        self.round_robin_index = 0
        self.model_keys = list(models.keys())
        
        # Pre-create queues and start processor tasks for all models
        for model_key in self.model_keys:
            self.queues[model_key] = asyncio.Queue()
            self.processors[model_key] = asyncio.create_task(self.process_queue(model_key))
            logger.info(f"Initialized queue for model: {model_key}")
    
    async def process_queue(self, model: str):
        queue = self.queues[model]
        model_instance = self.models[model]
        logger.info(f"Started queue processor for model: {model}")
        
        while True:
            try:
                req = await queue.get()
                if req is None:  # shutdown signal
                    logger.info(f"Shutdown signal received for model: {model}")
                    break
                    
                logger.debug(f"Processing request for model: {model}")
                result = await self._process_request(req, model_instance)
                req.future.set_result(result)
                queue.task_done()
            except Exception as e:
                logger.error(f"Error processing request for model {model}: {e}")
                if not req.future.done():
                    req.future.set_exception(e)
    
    async def _process_request(self, req: QueuedRequest, model_instance):
        try:
            # Keep duration computation in async path
            audio_info = sf.info(req.tmpname)
            duration_seconds = audio_info.duration
            
            # Offload blocking transcribe() call to thread
            transcribe_kwargs = {"source_lang": req.source_lang, "target_lang": req.target_lang}
            outputs = await asyncio.to_thread(
                model_instance.transcribe, 
                [req.tmpname], 
                **transcribe_kwargs
            )
            
            text = outputs[0].text if outputs else ""
            return {
                "text": text, 
                "model": req.model, 
                "source_lang": req.source_lang,
                "target_lang": req.target_lang, 
                "duration": duration_seconds
            }
        except Exception as e:
            logger.error(f"Error in _process_request: {e}")
            raise
    
    async def select_model(self, available_models):
        """Select a model using either round-robin or queue-size heuristic"""
        if self.use_round_robin:
            async with self.round_robin_lock:
                if not available_models:
                    raise ValueError("No available models")
                selected_model = available_models[self.round_robin_index % len(available_models)]
                self.round_robin_index += 1
                logger.debug(f"Round-robin selected model: {selected_model}")
                return selected_model
        else:
            # Use queue-size heuristic (no ephemeral Queue creation)
            selected_model = min(available_models, key=lambda m: self.queues[m].qsize())
            logger.debug(f"Queue-size heuristic selected model: {selected_model} (queue size: {self.queues[selected_model].qsize()})")
            return selected_model
    
    async def enqueue(self, model: str, tmpname: str, source_lang: str, target_lang: str):
        # Guard future creation with get_event_loop()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        req = QueuedRequest(model, tmpname, source_lang, target_lang, future)
        queue_size_before = self.queues[model].qsize()
        await self.queues[model].put(req)
        
        logger.info(f"Enqueued request for model {model}, queue size before: {queue_size_before}")
        return await future
    
    async def shutdown(self):
        """Gracefully shutdown all queues and processors"""
        logger.info("Shutting down queue manager...")
        
        # Send shutdown signals to all queues
        for model_key in self.model_keys:
            if model_key in self.queues:
                await self.queues[model_key].put(None)
        
        # Cancel all processor tasks
        for model_key in self.model_keys:
            if model_key in self.processors:
                task = self.processors[model_key]
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        
        logger.info("Queue manager shutdown complete")

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
parser.add_argument("--round-robin", action="store_true", help="Use round-robin model selection instead of queue-size heuristic")
args = parser.parse_args()

if not args.api_key:
    raise RuntimeError("INTERNAL_API_KEY must be provided via env or --api-key")

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
        logger.info("No GPUs found, loading model on CPU")
        # Load single model on CPU
        app.state.models[args.model] = load_model(args.model)
        app.state.model_names.append(args.model)
        logger.info(f"Loaded model {args.model} on CPU")
    else:
        logger.info(f"Found {num_gpus} GPU(s), loading {args.parallel_size} model instances")
        # Distribute models across available GPUs
        for i in range(args.parallel_size):
            gpu_id = i % num_gpus
            device = torch.device(f"cuda:{gpu_id}")
            model_key = f"{args.model}_gpu{gpu_id}_instance{i}"
            logger.info(f"Loading model instance {i} on {device}")
            app.state.models[model_key] = load_model(args.model, device)
            app.state.model_names.append(model_key)
            logger.info(f"Successfully loaded model {model_key} on device {device}")
    
    # Initialize queue manager with models and round-robin setting
    app.state.queue_manager = QueueManager(app.state.models, use_round_robin=args.round_robin)
    logger.info(f"Initialized queue manager with {len(app.state.models)} models, round_robin={args.round_robin}")
    
    try:
        yield
    finally:
        try:
            # Gracefully shutdown queue manager
            if hasattr(app.state, 'queue_manager'):
                await app.state.queue_manager.shutdown()
            
            # attempt to free model references on shutdown
            del app.state.models
            del app.state.model_names
            del app.state.queue_manager
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

app = FastAPI(title="NeMo OpenAI-compatible ASR", lifespan=lifespan)

def _check_auth(authorization: str):
    if authorization != f"Bearer {args.api_key}":
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
    if authorization != f"Bearer {args.api_key}":
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
    if authorization != f"Bearer {args.api_key}":
        REQUESTS.labels(endpoint='/v1/audio/transcriptions', status='401').inc()
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    model_name = model or args.model
    
    # Select a model instance (round-robin or queue-size based)
    available_models = app.state.model_names
    if not available_models:
        raise HTTPException(status_code=500, detail="No models available")
    
    # If a specific model was requested, check if it exists
    if model and model in app.state.models:
        selected_model = model
    # If requesting the base model name, select one of the instances
    elif model == args.model or not model:
        # Use the new selection logic (no ephemeral Queue creation)
        selected_model = await app.state.queue_manager.select_model(available_models)
    else:
        raise HTTPException(status_code=400, detail=f"Model {model} not found")

    # Fix temp file suffix fallback to ensure it works if filename is missing
    file_suffix = ".wav"
    if hasattr(file, 'filename') and file.filename:
        file_suffix = os.path.splitext(file.filename)[1] or ".wav"
    
    tmpfd, tmpname = tempfile.mkstemp(suffix=file_suffix)
    with os.fdopen(tmpfd, "wb") as f:
        f.write(await file.read())

    try:
        source_lang = source_lang or "es"
        target_lang = target_lang or "es"
        
        resp = await app.state.queue_manager.enqueue(selected_model, tmpname, source_lang, target_lang)
        REQUESTS.labels(endpoint='/v1/audio/transcriptions', status='200').inc()
        return JSONResponse(content=resp)
    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {e}")
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

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", args.host)
    port = int(os.environ.get("PORT", args.port))
    uvicorn.run(app, host=host, port=port)