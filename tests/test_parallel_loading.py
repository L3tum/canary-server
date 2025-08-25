#!/usr/bin/env python3
"""
Validation script to test parallel model loading
"""

import argparse
import torch
from nemo.collections.asr.models import ASRModel

def test_parallel_model_loading(model_name, parallel_size):
    """
    Test loading multiple instances of a model across GPUs
    
    Args:
        model_name (str): Name of the model to load
        parallel_size (int): Number of parallel instances to load
    """
    print(f"Testing parallel model loading for {model_name}")
    print(f"Parallel size: {parallel_size}")
    
    # Get available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus == 0:
        print("No GPUs available, testing on CPU")
        # Load single model on CPU
        model = ASRModel.from_pretrained(model_name=model_name)
        print(f"Successfully loaded model on CPU")
        return
    
    models = []
    for i in range(parallel_size):
        gpu_id = i % num_gpus
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Loading model instance {i} on {device}")
        
        # Load model (on CPU by default)
        model = ASRModel.from_pretrained(model_name=model_name)
        
        # Move to specific CUDA device
        model = model.to(device)
        models.append(model)
        print(f"Successfully loaded model instance {i} on {device}")
    
    print(f"Successfully loaded {parallel_size} model instances")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test parallel model loading")
    parser.add_argument("--model", default="nvidia/canary-1b-v2", 
                        help="Model name to load")
    parser.add_argument("--parallel-size", type=int, default=1,
                        help="Number of parallel model instances to load")
    
    args = parser.parse_args()
    
    test_parallel_model_loading(args.model, args.parallel_size)