#!/usr/bin/env python3
"""
Test script to demonstrate all available super-resolution models.
"""

import torch
import time
from src.model import *

def test_model(model, model_name, input_tensor):
    """Test a model with timing and memory usage."""
    print(f"\n=== Testing {model_name} ===")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        output = model(input_tensor)
        end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000  # ms
    
    print(f"Input shape:  {list(input_tensor.shape)}")
    print(f"Output shape: {list(output.shape)}")
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Calculate upscaling factor
    scale_h = output.shape[2] / input_tensor.shape[2]
    scale_w = output.shape[3] / input_tensor.shape[3]
    print(f"Upscaling factor: {scale_h:.1f}x{scale_w:.1f}")
    
    return output

def main():
    """Test all available models."""
    print("🔥 FIRE Model Testing Suite")
    print("=" * 50)
    
    # Create test input (batch=1, channels=3, height=64, width=64)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_tensor = torch.randn(1, 3, 64, 64).to(device)
    
    print(f"Device: {device}")
    print(f"Test input shape: {list(input_tensor.shape)}")
    
    # Test models
    models_to_test = [
        (EDSRLite(), "EDSRLite (Restoration)"),
        (SRCNN(), "SRCNN (Classic)"),
        (SRCNNPlusPlus(), "SRCNN++ (Enhanced)"),
        (EDSR(scale=2), "EDSR 2x"),
        (EDSR(scale=4), "EDSR 4x"),
        (RealESRGANLight(scale=4), "Real-ESRGAN Light 4x"),
    ]
    
    results = {}
    
    for model, name in models_to_test:
        try:
            model = model.to(device)
            output = test_model(model, name, input_tensor)
            results[name] = {
                'params': sum(p.numel() for p in model.parameters()),
                'output_shape': list(output.shape),
                'success': True
            }
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    successful_tests = sum(1 for r in results.values() if r.get('success', False))
    total_tests = len(results)
    
    print(f"Tests passed: {successful_tests}/{total_tests}")
    
    for name, result in results.items():
        if result.get('success'):
            params = result['params']
            shape = result['output_shape']
            print(f"✅ {name:<25} | {params:>8,} params | Output: {shape}")
        else:
            print(f"❌ {name:<25} | Failed: {result.get('error', 'Unknown error')}")
    
    print("\n🎉 Model testing complete!")

if __name__ == "__main__":
    main()