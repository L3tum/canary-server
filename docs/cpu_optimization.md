# CPU Optimization

When running the NeMo ASR server with multiple GPU instances, CPU contention can become a bottleneck. This document explains the CPU optimization features implemented in this server and how to configure them for optimal performance.

## Understanding the Bottleneck

In the original implementation, CPU-bound operations like audio preprocessing and model transcription were handled by the default thread pool used by `asyncio.to_thread()`. When using multiple GPU instances, this created contention as:

1. Multiple GPU instances were waiting for CPU-bound tasks to complete
2. The default thread pool size might not be optimal for the workload
3. Context switching and resource contention reduced overall throughput

With 4 GPU instances, performance could actually degrade compared to 1 or 2 instances due to this CPU bottleneck.

## Implemented Solutions

### 1. Dedicated Thread Pool

The server now uses a dedicated thread pool for CPU-bound operations, separate from the asyncio event loop and default thread pool. This ensures that:

- CPU-bound tasks don't block the main event loop
- Thread resources are managed specifically for preprocessing/transcription workloads
- Better isolation between different types of operations

### 2. Configurable Thread Pool Size

The thread pool size can be tuned based on your system's CPU resources and workload characteristics:

```bash
python -m src.nemo_openai_server --thread-pool-size 16
```

By default, the thread pool size is set to the number of CPU cores available, which is generally a good starting point.

### 3. CPU Affinity (Advanced)

For specialized deployments, CPU affinity binding can be used to:

- Pin worker threads to specific CPU cores
- Optimize cache locality
- Reduce context switching overhead

This feature can be disabled with:
```bash
python -m src.nemo_openai_server --disable-cpu-affinity
```

## Performance Tuning Recommendations

### Initial Setup

1. Start with the default settings (thread pool size = CPU cores)
2. Monitor CPU utilization during peak load
3. Adjust thread pool size based on observed utilization patterns

### Monitoring CPU Utilization

Use system monitoring tools to observe:

```bash
# Monitor CPU usage
htop

# Monitor per-core utilization
mpstat -P ALL 1

# Monitor context switches
vmstat 1
```

### Optimal Configurations

For different scenarios:

1. **Balanced Workload** (mixed CPU/GPU):
   ```
   --parallel-size 2 --thread-pool-size 8
   ```

2. **CPU-Bottlenecked Workload** (many short requests):
   ```
   --parallel-size 1 --thread-pool-size 16
   ```

3. **GPU-Bottlenecked Workload** (few long requests):
   ```
   --parallel-size 4 --thread-pool-size 8
   ```

## Benchmarking Your Configuration

To test the effectiveness of CPU optimizations:

1. Run a load test with your typical request pattern
2. Monitor both GPU and CPU utilization
3. Adjust configuration parameters
4. Re-run tests to compare performance

Example load test command:
```bash
# Using hey (https://github.com/rakyll/hey)
# With authentication (if API key configured):
hey -z 30s -c 10 -H "Authorization: Bearer your-api-key" \
# Without authentication:
# hey -z 30s -c 10 \
  -F "file=@test.wav" \
  -F "model=nvidia/canary-1b-v2" \
  http://localhost:8000/v1/audio/transcriptions
```

## Troubleshooting

### High CPU Usage

If CPU usage is consistently near 100%:

1. Reduce thread pool size:
   ```
   --thread-pool-size 4
   ```

2. Reduce parallel GPU instances:
   ```
   --parallel-size 1
   ```

### Poor GPU Utilization

If GPUs are underutilized:

1. Increase thread pool size to feed more work to GPUs:
   ```
   --thread-pool-size 16
   ```

2. Ensure sufficient batch sizes with:
   ```
   --max-batch-size 32 --max-wait-ms 50
   ```

### High Latency

If request latency is high:

1. Check if CPU or GPU is the bottleneck
2. Adjust thread pool size accordingly
3. Consider increasing `--max-wait-ms` to allow larger batches
```