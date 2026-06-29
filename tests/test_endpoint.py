#!/usr/bin/env python3
"""
Test script for the ASR endpoint
"""

import argparse
import os

import requests


def test_health_endpoint(api_key, host, port):
    """Test the health endpoint"""
    url = f"http://{host}:{port}/health"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print("✓ Health check passed")
            print(f"  Response: {response.json()}")
            return True
        else:
            print(f"✗ Health check failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Health check failed with exception: {e}")
        return False


def test_models_endpoint(api_key, host, port):
    """Test the models endpoint"""
    url = f"http://{host}:{port}/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            print("✓ Models endpoint check passed")
            print(f"  Available models: {[model['id'] for model in response.json()['data']]}")
            return True
        else:
            print(f"✗ Models endpoint check failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Models endpoint check failed with exception: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ASR endpoint")
    parser.add_argument(
        "--api-key", default=os.environ.get("INTERNAL_API_KEY", "your-api-key-here"), help="API key for authentication"
    )
    parser.add_argument("--host", default=os.environ.get("HOST", "localhost"), help="Server host")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Server port")

    args = parser.parse_args()

    print("Testing ASR endpoint...")
    print(f"Host: {args.host}:{args.port}")
    print()

    health_ok = test_health_endpoint(args.api_key, args.host, args.port)
    print()
    models_ok = test_models_endpoint(args.api_key, args.host, args.port)

    if health_ok and models_ok:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")
