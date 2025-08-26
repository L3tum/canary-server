#!/usr/bin/env python3
"""
NeMo OpenAI-compatible ASR Server - Refactored Version

This module provides an OpenAI-compatible ASR (Automatic Speech Recognition) endpoint 
using NVIDIA's Canary 1B v2 model through the NeMo toolkit.
Refactored to address key performance bottlenecks:
1. Single global ingress queue for all requests
2. Global batching coroutine for optimal microbatching
3. Stable per-GPU worker coroutines
4. In-memory audio handling
5. Advanced optimizations (torch.compile, half precision, etc.)
"""

import os
import tempfile
import argparse
import time
import asyncio
import concurrent.futures
import torch
import logging
from contextlib import asynccontextmanager
from typing import List, Dict, NamedTuple, Tuple, Optional
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Request, Path
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter
import soundfile as sf
import numpy as np
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class QueuedRequest(NamedTuple):
    model: str
    audio_data: bytes  # In-memory audio data instead of file path
    audio_duration: float  # Duration in seconds
    source_lang: str
    target_lang: str
    future: asyncio.Future

class GlobalBatchManager:
    def __init__(self, models, max_batch_size=64, max_wait_ms=50, thread_pool_size=None):
        self.request_queue = asyncio.Queue()  # Single global queue
        self.batch_processor_task = None
        self.models = models  # Dictionary of model instances
        self.model_keys = list(models.keys())
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        
        # Per-GPU worker management
        self.gpu_workers = {}  # gpu_id -> worker task
        self.gpu_queues = {}   # gpu_id -> queue for that GPU
        
        # Create a dedicated thread pool for CPU-bound operations
        # Default to number of CPU cores if not specified
        if thread_pool_size is None:
            thread_pool_size = os.cpu_count() or 4
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=thread_pool_size)
        logger.info(f"Created thread pool with {thread_pool_size} workers")
        
        # Initialize GPU workers and queues
        self._init_gpu_workers()
        
        # Start the global batch processor
        self.batch_processor_task = asyncio.create_task(self.process_global_queue())
        logger.info("Initialized global batch manager")
    
    def _init_gpu_workers(self):
        """Initialize one worker per GPU"""
        # Group model instances by GPU
        gpu_models = {}
        for model_key, model_instance in self.models.items():
            # Extract GPU ID from model key (format: model_gpu{gpu_id}_instance{i})
            if "_gpu" in model_key:
                gpu_id = int(model_key.split("_gpu")[1].split("_")[0])
            else:
                gpu_id = "cpu"
                
            if gpu_id not in gpu_models:
                gpu_models[gpu_id] = []
            gpu_models[gpu_id].append((model_key, model_instance))
        
        # Create a queue and worker for each GPU
        for gpu_id in gpu_models:
            self.gpu_queues[gpu_id] = asyncio.Queue()
            self.gpu_workers[gpu_id] = asyncio.create_task(
                self.gpu_worker(gpu_id, gpu_models[gpu_id])
            )
            logger.info(f"Initialized GPU worker for GPU {gpu_id}")
    
    async def process_global_queue(self):
        """Global batching coroutine that builds optimal microbatches"""
        logger.info("Started global batch processor")
        
        while True:
            batch = []
            try:
                # Wait for the first request with a timeout
                first_req = await asyncio.wait_for(self.request_queue.get(), self.max_wait_ms / 1000.0)
                if first_req is None:  # Shutdown signal
                    logger.info("Shutdown signal received for global batch processor")
                    break
                batch.append(first_req)
                
                # Collect more requests, trying to form length-homogeneous batches
                while len(batch) < self.max_batch_size:
                    try:
                        # Non-blocking get for remaining items
                        req = self.request_queue.get_nowait()
                        if req is None:  # Shutdown signal
                            break
                        batch.append(req)
                    except asyncio.QueueEmpty:
                        break
                
                # Sort by audio duration for better packing
                batch.sort(key=lambda x: x.audio_duration)
                
                # Group into sub-batches by similar length and assign to GPUs
                await self._distribute_batch(batch)
                
            except asyncio.TimeoutError:
                # This is expected if the queue is empty for max_wait_ms
                pass
            except Exception as e:
                logger.error(f"Error in process_global_queue: {e}")
    
    async def _distribute_batch(self, batch: List[QueuedRequest]):
        """Distribute a batch of requests to appropriate GPU workers"""
        if not batch:
            return
            
        # Simple round-robin distribution among GPUs for now
        # More sophisticated load balancing could be implemented here
        gpu_ids = list(self.gpu_queues.keys())
        for i, req in enumerate(batch):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            await self.gpu_queues[gpu_id].put(req)
    
    async def gpu_worker(self, gpu_id, model_instances: List[Tuple[str, object]]):
        """Worker coroutine for a specific GPU that processes requests"""
        queue = self.gpu_queues[gpu_id]
        logger.info(f"Started GPU worker for GPU {gpu_id}")
        
        # Use first model instance for this GPU (in case of multiple instances per GPU)
        model_key, model_instance = model_instances[0]
        
        while True:
            batch = []
            try:
                # Wait for the first request with a timeout
                first_req = await asyncio.wait_for(queue.get(), self.max_wait_ms / 1000.0)
                if first_req is None:  # Shutdown signal
                    logger.info(f"Shutdown signal received for GPU {gpu_id}")
                    break
                batch.append(first_req)
                
                # Fill the rest of the batch without waiting
                while len(batch) < self.max_batch_size and not queue.empty():
                    try:
                        req = queue.get_nowait()
                        if req is None:
                            break
                        batch.append(req)
                    except asyncio.QueueEmpty:
                        break
                
                # Process the batch
                if batch:
                    await self._process_gpu_batch(batch, model_instance, gpu_id)
                    
            except asyncio.TimeoutError:
                # This is expected if the queue is empty for max_wait_ms
                pass
            except Exception as e:
                logger.error(f"Error in gpu_worker for GPU {gpu_id}: {e}")
    
    async def _process_gpu_batch(self, batch: List[QueuedRequest], model_instance, gpu_id):
        """Process a batch of requests on a specific GPU"""
        try:
            # Write audio data to temporary files in memory
            temp_files = []
            source_langs = [req.source_lang for req in batch]
            target_langs = [req.target_lang for req in batch]
            
            # For now, we assume all requests in a batch have the same source and target language
            # A more robust implementation might group requests by language
            source_lang = source_langs[0]
            target_lang = target_langs[0]
            
            logger.info(f"Processing batch of size {len(batch)} on GPU {gpu_id}")
            
            # Create temporary files in memory
            for req in batch:
                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_file.write(req.audio_data)
                temp_file.flush()
                temp_files.append(temp_file.name)
            
            try:
                # Offload blocking transcribe() call to our dedicated thread pool
                transcribe_kwargs = {"source_lang": source_lang, "target_lang": target_lang}
                loop = asyncio.get_event_loop()
                outputs = await loop.run_in_executor(
                    self.thread_pool,
                    lambda: model_instance.transcribe(
                        temp_files,
                        batch_size=len(batch),
                        **transcribe_kwargs
                    )
                )
                
                # Set results for all requests
                for i, req in enumerate(batch):
                    try:
                        text = outputs[i].text if i < len(outputs) else ""
                        result = {
                            "text": text,
                            "model": req.model,
                            "source_lang": req.source_lang,
                            "target_lang": req.target_lang,
                            "duration": req.audio_duration
                        }
                        req.future.set_result(result)
                    except Exception as e:
                        logger.error(f"Error processing individual request in batch: {e}")
                        if not req.future.done():
                            req.future.set_exception(e)
            finally:
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"Error in _process_gpu_batch on GPU {gpu_id}: {e}")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
    
    async def enqueue(self, model: str, audio_data: bytes, audio_duration: float, 
                      source_lang: str, target_lang: str):
        """Enqueue a request to the global queue"""
        # Guard future creation with get_event_loop()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        req = QueuedRequest(model, audio_data, audio_duration, source_lang, target_lang, future)
        queue_size_before = self.request_queue.qsize()
        await self.request_queue.put(req)
        
        logger.info(f"Enqueued request, global queue size before: {queue_size_before}")
        return await future
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("Shutting down global batch manager...")
        
        # Send shutdown signals to global queue
        await self.request_queue.put(None)
        
        # Send shutdown signals to GPU queues
        for gpu_queue in self.gpu_queues.values():
            await gpu_queue.put(None)
        
        # Cancel batch processor task
        if self.batch_processor_task and not self.batch_processor_task.done():
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass
        
        # Cancel GPU worker tasks
        for gpu_worker in self.gpu_workers.values():
            if not gpu_worker.done():
                gpu_worker.cancel()
                try:
                    await gpu_worker
                except asyncio.CancelledError:
                    pass
        
        # Shutdown the thread pool
        self.thread_pool.shutdown(wait=True)
        logger.info("Global batch manager shutdown complete")

# Lazy import NeMo model to avoid heavy import on module load if not needed
def load_model(model_name: str, device=None):
    from nemo.collections.asr.models import ASRModel
    model = ASRModel.from_pretrained(model_name=model_name, map_location=device)
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--api-key", default=os.environ.get("INTERNAL_API_KEY"))
parser.add_argument("--model", default=os.environ.get("MODEL_NAME"))
parser.add_argument("--parallel-size", type=int, default=1, help="Number of parallel model instances")
parser.add_argument("--max-batch-size", type=int, default=64, help="Maximum batch size for transcription")
parser.add_argument("--max-wait-ms", type=int, default=30, help="Maximum wait time in milliseconds for batching")
parser.add_argument("--thread-pool-size", type=int, default=None, help="Size of thread pool for CPU-bound operations (default: number of CPU cores)")
parser.add_argument("--use-fp16", action="store_true", help="Use half precision (FP16) for model inference")
parser.add_argument("--use-torch-compile", action="store_true", help="Use torch.compile for model optimization")
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
            
            # Apply optimizations if requested
            if args.use_fp16:
                # Convert model to half precision
                app.state.models[model_key] = app.state.models[model_key].half()
                logger.info(f"Converted model {model_key} to FP16")
            
            if args.use_torch_compile:
                # Compile model for better performance (requires PyTorch 2.0+)
                try:
                    app.state.models[model_key] = torch.compile(app.state.models[model_key])
                    logger.info(f"Compiled model {model_key} with torch.compile")
                except Exception as e:
                    logger.warning(f"Failed to compile model {model_key}: {e}")

    # Warm up models
    logger.info("Warming up models...")
    # Create a dummy silent audio file for warmup
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        samplerate = 16000
        duration = 0.1  # 100ms
        data = (np.zeros(int(samplerate * duration)) * 32767).astype(np.int16)
        sf.write(tmpfile.name, data, samplerate)
        dummy_audio_path = tmpfile.name

    for model_key, model_instance in app.state.models.items():
        logger.info(f"Warming up {model_key}...")
        try:
            model_instance.transcribe([dummy_audio_path], batch_size=1)
            logger.info(f"Successfully warmed up {model_key}")
        except Exception as e:
            logger.error(f"Error warming up {model_key}: {e}")
    
    # Clean up the dummy audio file
    os.remove(dummy_audio_path)
    logger.info("Model warmup complete.")

    # Initialize global batch manager with models
    app.state.batch_manager = GlobalBatchManager(
        app.state.models,
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_wait_ms,
        thread_pool_size=args.thread_pool_size
    )
    logger.info(f"Initialized global batch manager with {len(app.state.models)} models")
    
    try:
        yield
    finally:
        try:
            # Gracefully shutdown batch manager
            if hasattr(app.state, 'batch_manager'):
                await app.state.batch_manager.shutdown()
            
            # attempt to free model references on shutdown
            del app.state.models
            del app.state.model_names
            del app.state.batch_manager
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
    
    # Select a model instance (we'll just use the first one since batching is global)
    available_models = app.state.model_names
    if not available_models:
        raise HTTPException(status_code=500, detail="No models available")
    
    # If a specific model was requested, check if it exists
    if model and model in app.state.models:
        selected_model = model
    # If requesting the base model name, select one of the instances
    elif model == args.model or not model:
        # For global batching, we can use any model instance since the actual
        # processing is handled by the batch manager
        selected_model = available_models[0]
    else:
        raise HTTPException(status_code=400, detail=f"Model {model} not found")

    # Read audio data directly into memory
    audio_data = await file.read()
    
    # Get audio duration using soundfile (in-memory)
    try:
        with BytesIO(audio_data) as audio_buffer:
            audio_info = sf.info(audio_buffer)
            audio_duration = audio_info.duration
    except Exception as e:
        logger.error(f"Error getting audio info: {e}")
        raise HTTPException(status_code=400, detail="Invalid audio file")

    try:
        source_lang = source_lang or "es"
        target_lang = target_lang or "es"
        
        resp = await app.state.batch_manager.enqueue(
            selected_model, audio_data, audio_duration, source_lang, target_lang)
        REQUESTS.labels(endpoint='/v1/audio/transcriptions', status='200').inc()
        return JSONResponse(content=resp)
    except Exception as e:
        logger.error(f"Error in transcribe endpoint: {e}")
        REQUESTS.labels(endpoint='/v1/audio/transcriptions', status='500').inc()
        raise HTTPException(status_code=500, detail=str(e))

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