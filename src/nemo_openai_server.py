#!/usr/bin/env python3
"""
NeMo OpenAI-compatible ASR Server - Refactored Version (Production Ready)

This module provides an OpenAI-compatible ASR (Automatic Speech Recognition) endpoint
using NVIDIA's Canary 1B v2 model through the NeMo toolkit.

Key features:
- Global batching with GPU workers for optimal throughput
- In-memory audio handling (no temporary files)
- FP16 precision and torch.compile optimizations
- Proper request timeouts and graceful error handling
- Structured logging with request ID tracing
- Comprehensive metrics (latency, queue depth, batch sizes)
- Enhanced health checks with GPU status
"""

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from io import BytesIO
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import slowapi
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from slowapi.util import get_remote_address


def get_client_ip(request: Request) -> str:
    """Get client IP address, respecting X-Forwarded-For headers for proxied requests."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Use the first IP (original client) from the X-Forwarded-For chain
        return forwarded_for.split(",")[0].strip()
    return get_remote_address(request)


DEFAULT_MODEL = "nvidia/canary-1b-v2"

# --- Request ID context variable for tracing ---
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


# --- Structured JSON logging with request ID ---
class JSONFormatter(logging.Formatter):
    """JSON log formatter with request ID support."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        req_id = request_id_var.get()
        if req_id:
            log_data["request_id"] = req_id
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }
        return json.dumps(log_data)


# Configure structured logging
class StructuredLogger(logging.Logger):
    """Logger that passes context to JSON formatter."""

    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
        record = super().makeRecord(name, level, fn, lno, msg, args, exc_info, func, extra, sinfo)
        record.request_id = request_id_var.get()
        return record


logging.setLoggerClass(StructuredLogger)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(JSONFormatter())
root_logger.handlers = [console_handler]
logger = logging.getLogger(__name__)

# --- Metrics ---
REQUESTS = Counter("nemo_requests_total", "Total number of requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("nemo_request_duration_seconds", "Request latency histogram", ["endpoint", "status"])
QUEUE_DEPTH = Gauge("nemo_queue_depth", "Current queue depth (number of pending requests)", ["type"])
BATCH_SIZE_HIST = Histogram(
    "nemo_batch_size", "Batch size distribution", ["gpu_id"], buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256]
)
MODEL_LOADED = Gauge("nemo_model_loaded", "Whether the model is loaded and ready", ["model"])

# --- Input validation ---
SUPPORTED_LANGUAGES = {
    "en",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "zh",
    "ja",
    "ko",
    "ru",
    "ar",
    "hi",
    "th",
    "vi",
    "tr",
    "pl",
    "nl",
    "sv",
    "fi",
    "da",
    "no",
    "el",
    "he",
    "cs",
    "sk",
    "hu",
    "ro",
    "bg",
    "hr",
    "sr",
    "sl",
    "et",
    "lv",
    "lt",
    "uk",
    "fa",
    "id",
    "ms",
    "sw",
    "tl",
    "auto",  # for source_lang if auto-detection is supported
}


class QueuedRequest(NamedTuple):
    model: str
    audio_data: bytes
    audio_duration: float
    source_lang: str
    target_lang: str
    future: asyncio.Future


class GlobalBatchManager:
    """Global batch manager that distributes requests across GPU workers."""

    def __init__(
        self,
        models: Dict[str, object],
        max_batch_size: int = 64,
        max_wait_ms: int = 50,
        thread_pool_size: Optional[int] = None,
    ):
        self.request_queue = asyncio.Queue()
        self.batch_processor_task = None
        self.models = models
        self.model_keys = list(models.keys())
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        # Per-GPU workers
        self.gpu_workers: Dict[str, asyncio.Task] = {}
        self.gpu_queues: Dict[str, asyncio.Queue] = {}

        # Thread pool for CPU-bound operations
        thread_pool_size = thread_pool_size or (os.cpu_count() or 4)
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=thread_pool_size, thread_name_prefix="asr-worker"
        )
        logger.info(f"Created thread pool with {thread_pool_size} workers")

        # Initialize GPU workers
        self._init_gpu_workers()

        # Start global batch processor
        self.batch_processor_task = asyncio.create_task(self.process_global_queue(), name="global-batch-processor")
        logger.info("Initialized global batch manager")

    def _init_gpu_workers(self):
        """Initialize one worker per GPU."""
        gpu_models: Dict[str, List[Tuple[str, object]]] = {}
        for model_key, model_instance in self.models.items():
            # Extract GPU ID from model key (format: model_gpu{gpu_id}_instance{i})
            if "_gpu" in model_key:
                parts = model_key.split("_gpu")
                gpu_id = int(parts[1].split("_")[0])
            else:
                gpu_id = "cpu"

            if gpu_id not in gpu_models:
                gpu_models[gpu_id] = []
            gpu_models[gpu_id].append((model_key, model_instance))

        for gpu_id in gpu_models:
            self.gpu_queues[gpu_id] = asyncio.Queue()
            self.gpu_workers[gpu_id] = asyncio.create_task(
                self.gpu_worker(gpu_id, gpu_models[gpu_id]), name=f"gpu-worker-{gpu_id}"
            )
            logger.info(f"Initialized GPU worker for GPU {gpu_id}")

    async def process_global_queue(self):
        """Global batching coroutine that builds optimal microbatches."""
        logger.info("Started global batch processor")

        while True:
            batch = []
            try:
                # Wait for first request with timeout
                first_req = await asyncio.wait_for(self.request_queue.get(), self.max_wait_ms / 1000.0)
                if first_req is None:
                    logger.info("Shutdown signal received for global batch processor")
                    break
                batch.append(first_req)

                # Collect more requests to form larger batches
                while len(batch) < self.max_batch_size:
                    try:
                        req = self.request_queue.get_nowait()
                        if req is None:
                            break
                        batch.append(req)
                    except asyncio.QueueEmpty:
                        break

                # Sort by audio duration for better packing
                batch.sort(key=lambda x: x.audio_duration)

                # Distribute homogeneous sub-batches to GPU workers
                await self._distribute_homogeneous_batches(batch)
                QUEUE_DEPTH.labels(type="global").set(self.request_queue.qsize())

            except asyncio.TimeoutError:
                pass  # Expected when queue is empty
            except Exception as e:
                logger.error(f"Error in process_global_queue: {e}", exc_info=True)

    async def _distribute_homogeneous_batches(self, batch: List[QueuedRequest]):
        """Partition requests by language pair and distribute sub-batches to GPU workers.

        This ensures each GPU only receives homogeneous batches, preventing
        "language pair mismatch" 500 errors when mixed-language requests arrive.
        """
        if not batch:
            return

        # Partition by (source_lang, target_lang)
        sub_batches: Dict[Tuple[str, str], List[QueuedRequest]] = {}
        for req in batch:
            key = (req.source_lang, req.target_lang)
            sub_batches.setdefault(key, []).append(req)

        # Round-robin distribute homogeneous sub-batches to GPUs.  The GPU queue
        # stores complete sub-batches (rather than individual requests) so the
        # worker cannot accidentally merge different language pairs back together.
        gpu_ids = list(self.gpu_queues.keys())
        for i, sub_batch in enumerate(sub_batches.values()):
            # Sort sub-batch by duration for better packing
            sub_batch.sort(key=lambda x: x.audio_duration)
            gpu_id = gpu_ids[i % len(gpu_ids)]
            await self.gpu_queues[gpu_id].put(sub_batch)

    async def gpu_worker(self, gpu_id: str, model_instances: List[Tuple[str, object]]):
        """Worker coroutine for a specific GPU that processes requests using round-robin model selection."""
        queue = self.gpu_queues[gpu_id]
        logger.info(
            f"Started GPU worker for GPU {gpu_id} with {len(model_instances)} model instance(s): "
            f"{[key for key, _ in model_instances]}"
        )

        rr_index = 0  # Round-robin index for multiple model instances on this GPU

        while True:
            batch = []
            try:
                # Wait for the next homogeneous sub-batch with timeout
                next_batch = await asyncio.wait_for(queue.get(), self.max_wait_ms / 1000.0)
                if next_batch is None:
                    logger.info(f"Shutdown signal received for GPU {gpu_id}")
                    break
                batch = next_batch

                # Select a model instance using round-robin
                model_key, model_instance = model_instances[rr_index % len(model_instances)]
                rr_index += 1

                # Process the already-homogeneous sub-batch
                if batch:
                    await self._process_gpu_batch(batch, model_instance, model_key)
                    QUEUE_DEPTH.labels(type="gpu").set(queue.qsize())

            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.error(f"Error in gpu_worker for GPU {gpu_id}: {e}", exc_info=True)

    async def _process_gpu_batch(self, batch: List[QueuedRequest], model_instance, model_key: str):
        """Process a batch of requests on a specific model instance."""
        start_time = time.time()
        try:
            source_langs = [req.source_lang for req in batch]
            target_langs = [req.target_lang for req in batch]

            # Assume all requests have same source/target lang (batch homogeneity)
            source_lang = source_langs[0]
            target_lang = target_langs[0]

            logger.info(f"Processing batch of size {len(batch)} on {model_key}")
            BATCH_SIZE_HIST.labels(gpu_id=model_key).observe(len(batch))

            # Validate batch homogeneity: all requests must have same language pair
            first_source = batch[0].source_lang
            first_target = batch[0].target_lang
            non_homogeneous = []
            for req in batch:
                if req.source_lang != first_source or req.target_lang != first_target:
                    non_homogeneous.append(req)

            # If batch is not homogeneous, return errors for mismatched requests
            if non_homogeneous:
                logger.warning(f"Non-homogeneous batch on {model_key}: {len(non_homogeneous)} mismatches")
                for req in non_homogeneous:
                    if not req.future.done():
                        req.future.set_exception(
                            RuntimeError(
                                f"Server error: Language pair mismatch in batch – "
                                f"batch uses source_lang={first_source}, target_lang={first_target}, "
                                f"but this request has source_lang={req.source_lang}, target_lang={req.target_lang}"
                            )
                        )
                batch = [req for req in batch if req not in non_homogeneous]
                if not batch:
                    return

            # Convert audio data to numpy arrays in memory, keeping successful reads aligned with outputs.
            audio_samples = []
            valid_requests = []
            for req in batch:
                try:
                    with BytesIO(req.audio_data) as bio:
                        samples, _ = sf.read(bio, dtype="float32")
                        audio_samples.append(samples)
                        valid_requests.append(req)
                except Exception as e:
                    logger.error(f"Error reading audio from request: {e}")
                    if not req.future.done():
                        req.future.set_exception(Exception(f"Failed to read audio: {e}"))

            if not audio_samples:
                return
            batch = valid_requests

            transcribe_kwargs = {"source_lang": source_lang, "target_lang": target_lang}
            loop = asyncio.get_event_loop()

            # Dynamic timeout based on longest audio duration (2x audio + 1 min overhead, max 10 min)
            max_audio_duration = max(req.audio_duration for req in batch)
            batch_timeout = min(max_audio_duration * 2 + 60, 600)

            # Run transcribe in thread pool with dynamic timeout
            try:
                outputs = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.thread_pool,
                        lambda: model_instance.transcribe(
                            audio_samples, batch_size=len(audio_samples), **transcribe_kwargs
                        ),
                    ),
                    timeout=batch_timeout,
                )

                # Set results
                output_idx = 0
                for req in batch:
                    if not req.future.done():
                        try:
                            text = outputs[output_idx].text if output_idx < len(outputs) else ""
                            result = {
                                "text": text,
                                "model": req.model,
                                "source_lang": req.source_lang,
                                "target_lang": req.target_lang,
                                "duration": req.audio_duration,
                            }
                            req.future.set_result(result)
                        except Exception as e:
                            logger.error(f"Error setting result for request: {e}", exc_info=True)
                            if not req.future.done():
                                req.future.set_exception(e)
                        finally:
                            output_idx += 1

                logger.info(f"Batch of {len(batch)} completed in {time.time() - start_time:.2f}s on {model_key}")

            except asyncio.TimeoutError:
                logger.error(f"Batch processing timed out after {batch_timeout:.0f}s on {model_key}")
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(Exception("Batch processing timed out"))

        except Exception as e:
            logger.error(f"Error in _process_gpu_batch on {model_key}: {e}", exc_info=True)
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    async def enqueue(self, model: str, audio_data: bytes, audio_duration: float, source_lang: str, target_lang: str):
        """Enqueue a request and wait for the result with timeout."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.get_running_loop()

        future = loop.create_future()
        req = QueuedRequest(model, audio_data, audio_duration, source_lang, target_lang, future)

        # Track queue depth metric
        queue_size = self.request_queue.qsize()
        QUEUE_DEPTH.labels(type="pending").set(queue_size + 1)
        await self.request_queue.put(req)

        try:
            # Wait for result with bounded timeout (audio + overhead, capped at 30 min)
            result = await asyncio.wait_for(
                future,
                timeout=min(audio_duration * 2 + 60, 1800),  # 2x audio + 1 min, max 30 min
            )
            return result
        except asyncio.TimeoutError:
            if not future.done():
                future.set_exception(Exception("Request timed out"))
            raise HTTPException(
                status_code=408, detail="Request timed out - audio too long or server overloaded"
            ) from None

    async def shutdown(self):
        """Gracefully shutdown all components."""
        logger.info("Shutting down global batch manager...")

        # Send shutdown signals
        await self.request_queue.put(None)
        for gpu_queue in self.gpu_queues.values():
            await gpu_queue.put(None)

        # Cancel tasks with timeout
        tasks = list(self.gpu_workers.values())
        if self.batch_processor_task:
            tasks.append(self.batch_processor_task)

        for task in tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to finish with timeout
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Some worker tasks didn't shut down in time")

        self.thread_pool.shutdown(wait=False)
        logger.info("Global batch manager shutdown complete")


# --- Model loading ---
def load_model(model_name: str, device=None):
    """Lazy import and load NeMo ASR model."""
    from nemo.collections.asr.models import ASRModel

    model = ASRModel.from_pretrained(model_name=model_name, map_location=device)
    return model


# --- Argument parsing (lazy, to avoid running on import) ---
def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a default."""
    value = os.environ.get(name)
    return int(value) if value else default


def _env_optional_int(name: str) -> Optional[int]:
    """Read an optional integer environment variable."""
    value = os.environ.get(name)
    return int(value) if value else None


def _parse_args():
    """Parse command-line arguments, tolerating unrelated runner arguments on import."""
    parser = argparse.ArgumentParser(description="NeMo OpenAI-compatible ASR Server")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=_env_int("PORT", 8000))
    parser.add_argument("--api-key", default=os.environ.get("INTERNAL_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--parallel-size", type=int, default=_env_int("PARALLEL_SIZE", 1))
    parser.add_argument("--max-batch-size", type=int, default=_env_int("MAX_BATCH_SIZE", 64))
    parser.add_argument("--max-wait-ms", type=int, default=_env_int("MAX_WAIT_MS", 30))
    parser.add_argument("--thread-pool-size", type=int, default=_env_optional_int("THREAD_POOL_SIZE"))
    parser.add_argument("--rate-limit", default=os.environ.get("RATE_LIMIT", "30/minute"))
    parser.add_argument("--use-fp16", action="store_true", default=os.environ.get("USE_FP16", "").lower() == "true")
    parser.add_argument(
        "--use-torch-compile",
        action="store_true",
        default=os.environ.get("USE_TORCH_COMPILE", "").lower() == "true",
    )
    parsed_args, unknown_args = parser.parse_known_args()
    if unknown_args and os.path.basename(sys.argv[0]) not in {"pytest", "uvicorn"}:
        logger.debug(f"Ignoring unknown command-line arguments: {unknown_args}")
    return parsed_args


args = _parse_args()

if not args.api_key:
    logger.warning(
        "No API key configured – all endpoints are unauthenticated. "
        "Set INTERNAL_API_KEY for production security."
    )
else:
    logger.info("API key configured – authentication enabled")


# --- Lifespan: Load models before serving ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = {}
    app.state.model_names = []
    app.state.ready = False

    # Determine device
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logger.info("No GPUs found, loading model on CPU")
        model = load_model(args.model)
        app.state.models[args.model] = model
        app.state.model_names.append(args.model)
    else:
        logger.info(f"Found {num_gpus} GPU(s), loading {args.parallel_size} instances")
        for i in range(args.parallel_size):
            gpu_id = i % num_gpus
            device = torch.device(f"cuda:{gpu_id}")
            model_key = f"{args.model}_gpu{gpu_id}_instance{i}"
            logger.info(f"Loading {model_key} on {device}")
            model = load_model(args.model, device)

            # Apply optimizations
            if args.use_fp16:
                model = model.half()
                logger.info(f"Converted {model_key} to FP16")
            if args.use_torch_compile:
                try:
                    model = torch.compile(model)
                    logger.info(f"Compiled {model_key} with torch.compile")
                except Exception as e:
                    logger.warning(f"Failed to compile {model_key}: {e}")

            app.state.models[model_key] = model
            app.state.model_names.append(model_key)

    # Warm up models
    logger.info("Warming up models...")
    samplerate = 16000
    duration = 0.1
    dummy_audio = np.zeros(int(samplerate * duration), dtype=np.float32)

    for model_key, model_instance in app.state.models.items():
        try:
            model_instance.transcribe([dummy_audio], batch_size=1)
            MODEL_LOADED.labels(model=model_key).set(1)
            logger.info(f"Warmed up {model_key}")
        except Exception as e:
            MODEL_LOADED.labels(model=model_key).set(0)
            logger.error(f"Failed to warm up {model_key}: {e}")

    app.state.batch_manager = GlobalBatchManager(
        app.state.models,
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_wait_ms,
        thread_pool_size=args.thread_pool_size,
    )
    app.state.ready = True
    logger.info(f"Server ready with {len(app.state.model_names)} models")

    try:
        yield
    finally:
        try:
            if hasattr(app.state, "batch_manager"):
                await app.state.batch_manager.shutdown()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# --- FastAPI app ---
app = FastAPI(title="NeMo OpenAI-compatible ASR", lifespan=lifespan, version="1.0.0")

# Add CORS middleware (allow all origins, no credentials to avoid browser security warning)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiter with IP-based key (supports X-Forwarded-For for reverse proxies)
limiter = slowapi.Limiter(key_func=get_client_ip)
app.state.limiter = limiter


def _check_auth(authorization: str):
    """Check authorization header. If no API key is configured, skip auth."""
    if args.api_key and authorization != f"Bearer {args.api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _model_obj(name: str) -> Dict:
    """Create model object for /v1/models endpoint."""
    return {"id": name, "object": "model", "created": int(time.time()), "owned_by": "nemo", "permission": []}


def _available_models() -> List[str]:
    """Return list of available models."""
    names = set()
    if args.model:
        names.add(args.model)
    loaded = getattr(app.state, "model_names", [])
    names.update(loaded)
    return sorted([n for n in names if n])


def validate_language(lang: str, field: str) -> str:
    """Validate and return a language code. This is a FastAPI utility that may raise HTTPException."""
    normalized = lang.lower()
    if normalized not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400, detail=f"Invalid {field}: '{lang}'. Supported languages: {sorted(SUPPORTED_LANGUAGES)}"
        )
    if field == "target_lang" and normalized == "auto":
        raise HTTPException(status_code=400, detail="Invalid target_lang: 'auto' is only supported for source_lang")
    return normalized


# --- Middleware for request ID and logging ---
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request for tracing."""
    request_id = str(uuid.uuid4())
    token = request_id_var.set(request_id)
    request.state.request_id = request_id
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        request_id_var.reset(token)


# --- Endpoints ---


@app.get("/health")
async def health():
    """Simple health check for monitoring (no auth required). Returns 503 if models not ready."""
    if not getattr(app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Service starting, models not loaded")
    return {"status": "ok"}


@app.get("/healthz")
async def healthz(
    authorization: str = Header(None),
    request: Request = None,
):
    """Health check endpoint with detailed GPU status."""
    if args.api_key and authorization != f"Bearer {args.api_key}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    gpu_info = {}
    if torch.cuda.is_available():
        for gpu in range(torch.cuda.device_count()):
            try:
                alloc = torch.cuda.memory_allocated(gpu) / 1e9  # GB
                reserved = torch.cuda.memory_reserved(gpu) / 1e9
                name = torch.cuda.get_device_name(gpu)
                gpu_info[gpu] = {
                    "name": name,
                    "memory_allocated_gb": f"{alloc:.2f}",
                    "memory_reserved_gb": f"{reserved:.2f}",
                }
            except Exception:
                gpu_info[gpu] = "error"

    models_loaded = len(app.state.model_names) if hasattr(app.state, "model_names") else 0
    ready = app.state.ready if hasattr(app.state, "ready") else False

    queue_depth = (
        app.state.batch_manager.request_queue.qsize()
        if hasattr(app.state, "batch_manager") and hasattr(app.state.batch_manager, "request_queue")
        else 0
    )

    return {
        "status": "ok",
        "request_id": request.state.request_id,
        "ready": ready,
        "models_loaded": models_loaded,
        "gpu_count": torch.cuda.device_count(),
        "gpu_info": gpu_info,
        "queue_depth": queue_depth,
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint (no auth)."""
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/audio/transcriptions")
@limiter.limit(args.rate_limit)  # Default: 30 requests per minute per IP
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(None),
    source_lang: str = Form("en"),
    target_lang: str = Form("en"),
    authorization: str = Header(None),
):
    """Transcribe audio file (OpenAI-compatible endpoint)."""
    if args.api_key and authorization != f"Bearer {args.api_key}":
        REQUESTS.labels(endpoint="/v1/audio/transcriptions", status="401").inc()
        raise HTTPException(status_code=401, detail="Unauthorized")

    start_time = time.time()

    try:
        # Select model
        available_models = app.state.model_names
        if not available_models:
            raise HTTPException(status_code=500, detail="No models available")

        if model and model in app.state.models:
            selected_model = model
        elif model == args.model or not model:
            selected_model = available_models[0]
        else:
            raise HTTPException(status_code=400, detail=f"Model {model} not found")

        # Read and validate audio data in memory
        audio_data = await file.read()
        max_file_size = 500 * 1024 * 1024  # 500MB limit
        if len(audio_data) > max_file_size:
            raise HTTPException(status_code=413, detail="File too large (max 500MB)")

        # Validate audio format and get duration
        try:
            with BytesIO(audio_data) as bio:
                audio_info = sf.info(bio)
                audio_duration = audio_info.duration
                # Check sample rate and channels
                if audio_info.samplerate > 96000:
                    raise HTTPException(
                        status_code=400, detail=f"Sample rate {audio_info.samplerate} too high (max 96kHz)"
                    )
                if audio_duration > 1800:  # 30 minutes max
                    raise HTTPException(status_code=400, detail="Audio too long (max 30 minutes)")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}") from e

        # Validate language codes
        source_lang = validate_language(source_lang, "source_lang")
        target_lang = validate_language(target_lang, "target_lang")

        # Enqueue and process
        resp = await app.state.batch_manager.enqueue(
            selected_model, audio_data, audio_duration, source_lang, target_lang
        )
        REQUESTS.labels(endpoint="/v1/audio/transcriptions", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/v1/audio/transcriptions", status="200").observe(time.time() - start_time)

        return JSONResponse(content=resp)

    except HTTPException:
        # Re-raise HTTP exceptions with timing
        raise
    except Exception as e:
        REQUESTS.labels(endpoint="/v1/audio/transcriptions", status="500").inc()
        REQUEST_LATENCY.labels(endpoint="/v1/audio/transcriptions", status="500").observe(time.time() - start_time)
        logger.error(f"Error in transcribe endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    """List available models (OpenAI-compatible)."""
    try:
        _check_auth(authorization)
    except HTTPException:
        REQUESTS.labels(endpoint="/v1/models", status="401").inc()
        raise

    models = [_model_obj(name) for name in _available_models()]
    resp = {"object": "list", "data": models}
    REQUESTS.labels(endpoint="/v1/models", status="200").inc()
    return JSONResponse(content=resp)


def main():
    """Run the ASR server with uvicorn."""
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
