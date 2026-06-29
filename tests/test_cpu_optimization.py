#!/usr/bin/env python3
"""
Test script to validate CPU optimization improvements
"""

import argparse
import asyncio
import logging
import os
import sys
import time

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def transcribe_file(session, url, api_key, file_path, model=None):
    """Transcribe a single file"""
    with open(file_path, "rb") as f:
        data = aiohttp.FormData()
        data.add_field("file", f, filename=os.path.basename(file_path))
        if model:
            data.add_field("model", model)

        headers = {"Authorization": f"Bearer {api_key}"}

        start_time = time.time()
        async with session.post(url, data=data, headers=headers) as resp:
            response_time = time.time() - start_time
            if resp.status == 200:
                result = await resp.json()
                return {
                    "status": "success",
                    "response_time": response_time,
                    "text_length": len(result.get("text", "")),
                    "duration": result.get("duration", 0),
                }
            else:
                text = await resp.text()
                return {"status": "error", "response_time": response_time, "error": f"HTTP {resp.status}: {text}"}


async def run_load_test(base_url, api_key, file_path, concurrent_requests, total_requests, model=None):
    """Run a load test with specified concurrency"""
    url = f"{base_url}/v1/audio/transcriptions"

    # Track metrics
    results = []
    start_time = time.time()

    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrent_requests)

    async def limited_transcribe():
        async with semaphore:
            return await transcribe_file(session, url, api_key, file_path, model)

    async with aiohttp.ClientSession() as session:
        # Create tasks for all requests
        tasks = [limited_transcribe() for _ in range(total_requests)]

        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

    total_time = time.time() - start_time

    # Process results
    successful_requests = 0
    total_response_time = 0
    errors = 0

    for result in results:
        if isinstance(result, Exception):
            errors += 1
            logger.error(f"Request failed with exception: {result}")
        elif result["status"] == "success":
            successful_requests += 1
            total_response_time += result["response_time"]
        else:
            errors += 1
            logger.error(f"Request failed: {result['error']}")

    # Calculate metrics
    requests_per_second = total_requests / total_time
    avg_response_time = total_response_time / successful_requests if successful_requests > 0 else 0

    print("\n=== Load Test Results ===")
    print(f"Total requests: {total_requests}")
    print(f"Concurrent requests: {concurrent_requests}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Requests per second: {requests_per_second:.2f}")
    print(f"Average response time: {avg_response_time:.2f} seconds")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests: {errors}")

    return {
        "total_requests": total_requests,
        "concurrent_requests": concurrent_requests,
        "total_time": total_time,
        "requests_per_second": requests_per_second,
        "average_response_time": avg_response_time,
        "successful_requests": successful_requests,
        "failed_requests": errors,
    }


def main():
    parser = argparse.ArgumentParser(description="Test CPU optimization improvements")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument("--file", required=True, help="Audio file to transcribe")
    parser.add_argument("--concurrent", type=int, default=5, help="Concurrent requests")
    parser.add_argument("--total", type=int, default=20, help="Total requests")
    parser.add_argument("--model", help="Model to use")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        sys.exit(1)

    base_url = f"http://{args.host}:{args.port}"

    logger.info("Starting load test...")
    logger.info(f"Server: {base_url}")
    logger.info(f"File: {args.file}")
    logger.info(f"Concurrent requests: {args.concurrent}")
    logger.info(f"Total requests: {args.total}")

    # Run the load test
    results = asyncio.run(run_load_test(base_url, args.api_key, args.file, args.concurrent, args.total, args.model))

    # Print summary
    print("\n=== Summary ===")
    print(f"Throughput: {results['requests_per_second']:.2f} requests/second")
    print(f"Average latency: {results['average_response_time']:.2f} seconds")

    if results["failed_requests"] > 0:
        print(f"⚠️  {results['failed_requests']} requests failed")
        sys.exit(1)
    else:
        print("✅ All requests successful")


if __name__ == "__main__":
    main()
