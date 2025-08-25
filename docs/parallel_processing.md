# Parallel Processing with Multiple GPU Support

The NeMo OpenAI-compatible ASR server supports parallel processing by loading multiple instances of the same model across available GPUs. This feature improves throughput by allowing simultaneous transcription requests to be processed on different GPU devices.

## How It Works

When starting the server with the `--parallel-size` argument, the server will:

1. Detect available GPU devices using PyTorch
2. Load the specified number of model instances
3. Distribute these instances across available GPUs in a round-robin fashion
4. Process incoming requests using a queue system that balances load across all instances

## Command Line Usage

Start the server with parallel processing:

```bash
python -m src.nemo_openai_server \
  --api-key your-secret-api-key \
  --model nvidia/canary-1b-v2 \
  --parallel-size 4 \
  --host 0.0.0.0 \
  --port 8000
```

This example will load 4 instances of the model distributed across available GPUs.

## Environment Variable Configuration

You can also configure parallel processing using environment variables:

```bash
export INTERNAL_API_KEY=\"your-secret-api-key\"
export MODEL_NAME=\"nvidia/canary-1b-v2\"
export PARALLEL_SIZE=4
python -m src.nemo_openai_server
```

## GPU Distribution Strategy

The server uses a round-robin distribution strategy:

- If you have 2 GPUs (cuda:0, cuda:1) and request 4 parallel instances:
  - Instance 0 → cuda:0
  - Instance 1 → cuda:1
  - Instance 2 → cuda:0
  - Instance 3 → cuda:1

- If you have 4 GPUs and request 2 parallel instances:
  - Instance 0 → cuda:0
  - Instance 1 → cuda:1

## Performance Considerations

1. **Memory Requirements**: Each model instance consumes GPU memory. Ensure you have sufficient GPU memory for the requested number of instances.

2. **Optimal Sizing**: The optimal number of parallel instances depends on:
   - GPU memory capacity
   - Model size
   - Expected concurrent request volume

3. **CPU Bottlenecks**: While GPU parallelization helps, the CPU may become a bottleneck with high parallelization levels.

## Monitoring Parallel Instances

The server provides metrics at the `/metrics` endpoint that include information about request distribution across model instances.

## Fallback to CPU

If no GPUs are available, the server will load a single model instance on CPU regardless of the `--parallel-size` setting.