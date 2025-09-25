#!/usr/bin/env python3
"""
FIRE Model Benchmark Script

This script benchmarks all available super-resolution models to help users
choose the best model for their use case based on parameter count, memory usage,
and inference speed.
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import gc

from src.model import MODEL_REGISTRY, MODEL_PARAMS


def benchmark_model(model_name: str, model_class: type, input_size: Tuple[int, int] = (64, 64),
                   scale: int = 4, num_runs: int = 10, device: str = 'cpu') -> Dict:
    """Benchmark a single model."""
    
    print(f"\nBenchmarking {model_name}...")
    
    # Create model
    try:
        if 'swinir' in model_name.lower():
            model = model_class(upscale=scale)
        elif 'edsr_lite' in model_name.lower():
            model = model_class()  # No upscaling for restoration model
        else:
            model = model_class(scale=scale)
    except Exception as e:
        print(f"  ❌ Failed to create model: {e}")
        return {}
    
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create input
    if 'edsr_lite' in model_name.lower():
        # Restoration model - same size input/output
        input_tensor = torch.randn(1, 3, *input_size, device=device).clamp(0, 1)
        expected_output_size = input_size
    else:
        # Super-resolution model
        input_tensor = torch.randn(1, 3, *input_size, device=device).clamp(0, 1)
        expected_output_size = (input_size[0] * scale, input_size[1] * scale)
    
    # Warmup
    try:
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_tensor)
        torch.cuda.synchronize() if device == 'cuda' else None
    except Exception as e:
        print(f"  ❌ Failed during warmup: {e}")
        return {}
    
    # Benchmark inference time
    times = []
    memory_used = 0
    
    try:
        for i in range(num_runs):
            if device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = model(input_tensor)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            
            if device == 'cuda' and i == 0:  # Measure memory on first run
                memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            # Verify output shape
            if i == 0:
                expected_shape = (1, 3, *expected_output_size)
                if output.shape != expected_shape:
                    print(f"  ⚠️  Unexpected output shape: {output.shape}, expected: {expected_shape}")
    
    except Exception as e:
        print(f"  ❌ Failed during benchmark: {e}")
        return {}
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    # FPS calculation
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    results = {
        'parameters': total_params,
        'trainable_parameters': trainable_params,
        'avg_inference_time_ms': avg_time * 1000,
        'min_inference_time_ms': min_time * 1000,
        'max_inference_time_ms': max_time * 1000,
        'fps': fps,
        'memory_mb': memory_used,
        'input_shape': input_tensor.shape,
        'output_shape': output.shape if 'output' in locals() else None,
    }
    
    print(f"  ✅ Parameters: {total_params:,}")
    print(f"  ✅ Avg time: {avg_time*1000:.2f}ms ({fps:.1f} FPS)")
    if device == 'cuda':
        print(f"  ✅ Memory: {memory_used:.1f}MB")
    
    return results


def main():
    """Run benchmarks for all models."""
    
    print("🔥 FIRE Model Benchmark")
    print("=" * 50)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Benchmark settings
    input_size = (64, 64)  # Standard test size
    scale = 4
    num_runs = 10
    
    print(f"Input size: {input_size}")
    print(f"Scale factor: {scale}x")
    print(f"Runs per model: {num_runs}")
    
    # Models to benchmark
    models_to_test = [
        'edsr_lite', 'edsr_small', 'edsr_medium', 'edsr_large',
        'real_esrgan_small', 'real_esrgan_medium', 'real_esrgan_large',
        'swinir_small', 'swinir_medium', 'swinir_large'
    ]
    
    results = {}
    
    for model_name in models_to_test:
        if model_name in MODEL_REGISTRY:
            model_class = MODEL_REGISTRY[model_name]
            result = benchmark_model(
                model_name, model_class, input_size, scale, num_runs, device
            )
            if result:
                results[model_name] = result
        else:
            print(f"\n❌ Model {model_name} not found in registry")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 80)
    
    if results:
        print(f"{'Model':<18} {'Params':<12} {'Time(ms)':<10} {'FPS':<8} {'Memory(MB)':<12}")
        print("-" * 78)
        
        # Sort by parameter count
        sorted_results = sorted(results.items(), key=lambda x: x[1]['parameters'])
        
        for model_name, result in sorted_results:
            params = f"{result['parameters']:,}"
            if len(params) > 11:
                params = f"{result['parameters']/1e6:.1f}M"
            
            avg_time = result['avg_inference_time_ms']
            fps = result['fps']
            memory = result['memory_mb']
            
            print(f"{model_name:<18} {params:<12} {avg_time:<10.2f} {fps:<8.1f} {memory:<12.1f}")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("  • Fastest: edsr_small")
    print("  • Balanced: real_esrgan_medium")  
    print("  • Best Quality: edsr_large")
    print("  • Real-world Images: real_esrgan_large")
    print("  • Transformer: swinir_medium")
    print("  • Restoration (1x): edsr_lite")


if __name__ == "__main__":
    main()