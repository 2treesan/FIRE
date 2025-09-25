#!/usr/bin/env python3
"""
Example usage of FIRE super-resolution models.
Demonstrates how to use different models for various SR tasks.
"""

import torch
import numpy as np
from PIL import Image
from src.model import SRCNN, EDSR, RealESRGANLight

def create_sample_image(size=(64, 64)):
    """Create a sample RGB image for testing."""
    # Create a simple gradient pattern
    h, w = size
    r = np.linspace(0, 1, w)[None, :].repeat(h, axis=0)
    g = np.linspace(0, 1, h)[:, None].repeat(w, axis=1) 
    b = np.ones((h, w)) * 0.5
    
    # Add some texture
    x, y = np.meshgrid(np.linspace(0, 4*np.pi, w), np.linspace(0, 4*np.pi, h))
    pattern = 0.1 * np.sin(x) * np.cos(y)
    
    r = np.clip(r + pattern, 0, 1)
    g = np.clip(g + pattern, 0, 1)
    b = np.clip(b + pattern, 0, 1)
    
    image = np.stack([r, g, b], axis=2)
    return (image * 255).astype(np.uint8)

def tensor_to_pil(tensor):
    """Convert tensor to PIL image."""
    # tensor: [C, H, W] in [0, 1]
    array = (tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)

def pil_to_tensor(image):
    """Convert PIL image to tensor."""
    # Convert to tensor: [C, H, W] in [0, 1]
    array = np.array(image).astype(np.float32) / 255.0
    if len(array.shape) == 3:
        return torch.from_numpy(array).permute(2, 0, 1)
    else:
        return torch.from_numpy(array).unsqueeze(0)

def main():
    """Demonstrate different SR models."""
    print("🔥 FIRE Super-Resolution Example")
    print("=" * 40)
    
    # Create sample low-resolution image
    lr_array = create_sample_image((64, 64))
    lr_pil = Image.fromarray(lr_array)
    lr_tensor = pil_to_tensor(lr_pil).unsqueeze(0)  # Add batch dimension
    
    print(f"Input image size: {lr_pil.size}")
    print(f"Input tensor shape: {lr_tensor.shape}")
    
    # Test different models
    models = {
        "SRCNN (Same Size)": SRCNN(),
        "EDSR 2x Upscaling": EDSR(scale=2),
        "EDSR 4x Upscaling": EDSR(scale=4),
        "Real-ESRGAN Light 4x": RealESRGANLight(scale=4),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Testing {name} ---")
        
        model.eval()
        with torch.no_grad():
            # For SRCNN, we need to bicubic upsample the input first
            if "SRCNN" in name:
                # Bicubic upsample to target size (let's say 4x for demonstration)
                target_size = (lr_tensor.shape[2] * 4, lr_tensor.shape[3] * 4)
                input_tensor = torch.nn.functional.interpolate(
                    lr_tensor, size=target_size, mode='bicubic', align_corners=False
                )
            else:
                input_tensor = lr_tensor
            
            output = model(input_tensor)
        
        print(f"  Input shape:  {list(input_tensor.shape)}")
        print(f"  Output shape: {list(output.shape)}")
        print(f"  Parameters:   {sum(p.numel() for p in model.parameters()):,}")
        
        # Convert back to PIL for potential saving
        output_pil = tensor_to_pil(output[0])  # Remove batch dimension
        results[name] = output_pil
        
        scale_factor = output.shape[2] / lr_tensor.shape[2]
        print(f"  Effective scale: {scale_factor:.1f}x")
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Summary")
    print("=" * 40)
    
    for name, pil_image in results.items():
        print(f"{name:<25} | Output size: {pil_image.size}")
    
    print("\n🎯 Usage Recommendations:")
    print("• SRCNN: Educational purposes, understanding SR basics")
    print("• EDSR 2x: High-quality 2x upscaling for clean images")  
    print("• EDSR 4x: High-quality 4x upscaling for clean images")
    print("• Real-ESRGAN: Best for real-world degraded images")
    
    print("\n✨ Example complete! Models are ready for training.")

if __name__ == "__main__":
    main()