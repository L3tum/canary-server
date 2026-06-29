#!/usr/bin/env python3
"""
Benchmark script for testing parallel processing performance
"""

import argparse
import asyncio
import os
import time
from typing import List

import aiohttp


async def transcribe_file(session: aiohttp.ClientSession, url: str, api_key: str, file_path: str, model: str) -> dict:
    """Transcribe a single audio file"""
    headers = {"Authorization": f"Bearer {api_key}"}

    with open(file_path, "rb") as f:
        data = aiohttp.FormData()
        data.add_field("file", f, filename=os.path.basename(file_path))
        data.add_field("model", model)
        data.add_field("source_lang", "es")
        data.add_field("target_lang", "es")

        async with session.post(url, headers=headers, data=data) as response:
            return await response.json()


async def benchmark_parallel_processing(
    api_key: str, host: str, port: int, file_paths: List[str], model: str, concurrent_requests: int
) -> None:
    """Benchmark parallel processing performance"""
    url = f"http://{host}:{port}/v1/audio/transcriptions"

    # Create a connector with connection limits
    connector = aiohttp.TCPConnector(limit=concurrent_requests, limit_per_host=concurrent_requests)

    async with aiohttp.ClientSession(connector=connector) as session:
        # Create tasks for all files
        start_time = time.time()

        tasks = []
        for i, _file_path in enumerate(file_paths):
            # Cycle through files if we have more requests than files
            file_to_use = file_paths[i % len(file_paths)]
            task = transcribe_file(session, url, api_key, file_to_use, model)
            tasks.append(task)

        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()

        # Calculate statistics
        successful_requests = sum(1 for r in results if not isinstance(r, Exception))
        total_time = end_time - start_time
        requests_per_second = successful_requests / total_time if total_time > 0 else 0

        print("Benchmark Results:")
        print(f"  Total requests: {len(file_paths)}")
        print(f"  Concurrent requests: {concurrent_requests}")
        print(f"  Successful requests: {successful_requests}")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Requests per second: {requests_per_second:.2f}")

        # Show any errors
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            print(f"  Errors: {len(errors)}")
            for i, error in enumerate(errors[:5]):  # Show first 5 errors
                print(f"    Error {i + 1}: {error}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark parallel processing performance")
    parser.add_argument(
        "--api-key", default=os.environ.get("INTERNAL_API_KEY", "your-api-key-here"), help="API key for authentication"
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "localhost"), help="Server host")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Server port")
    parser.add_argument("--files", nargs="+", required=True, help="Audio files to transcribe")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "nvidia/canary-1b-v2"), help="Model to use")
    parser.add_argument("--concurrent", type=int, default=4, help="Number of concurrent requests")

    args = parser.parse_args()

    print("Starting benchmark...")
    print(f"Server: {args.host}:{args.port}")
    print(f"Files: {len(args.files)}")
    print(f"Concurrent requests: {args.concurrent}")
    print()

    asyncio.run(
        benchmark_parallel_processing(args.api_key, args.host, args.port, args.files, args.model, args.concurrent)
    )


if __name__ == "__main__":
    main()
