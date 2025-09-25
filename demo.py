#!/usr/bin/env python3
"""
FIRE Demo Script

Demonstrates the super-resolution capabilities of different models
with synthetic test images.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.model import create_model, get_recommended_model


def create_test_image(size: int = 64) -> torch.Tensor:
    """Create a synthetic test image with various patterns."""
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create a complex pattern with multiple frequencies
    pattern1 = torch.sin(5 * torch.pi * X) * torch.cos(5 * torch.pi * Y)
    pattern2 = torch.sin(10 * torch.pi * X) * torch.cos(3 * torch.pi * Y)
    pattern3 = torch.exp(-(X**2 + Y**2) * 3)  # Gaussian
    
    # Combine patterns
    image = pattern1 * 0.3 + pattern2 * 0.3 + pattern3 * 0.4
    
    # Convert to RGB
    image = image.unsqueeze(0).repeat(3, 1, 1)
    
    # Add some color variation
    image[0] *= 1.2  # More red
    image[1] *= 0.8  # Less green
    image[2] *= 1.1  # Slightly more blue
    
    # Normalize to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    
    return image.unsqueeze(0)  # Add batch dimension


def create_degraded_image(hr_image: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """Create a degraded low-resolution version of the image."""
    # Simulate degradation: downsample -> blur -> add noise
    lr_size = hr_image.shape[-1] // scale
    
    # Downsample
    lr_image = F.interpolate(hr_image, size=(lr_size, lr_size), mode='bicubic', align_corners=False)
    
    # Add slight blur
    blur_kernel = torch.ones(1, 1, 3, 3) / 9.0
    blur_kernel = blur_kernel.repeat(3, 1, 1, 1)
    lr_image = F.conv2d(lr_image, blur_kernel, padding=1, groups=3)
    
    # Add small amount of noise
    noise = torch.randn_like(lr_image) * 0.02
    lr_image = lr_image + noise
    
    return lr_image.clamp(0, 1)


def demonstrate_models():
    """Demonstrate different super-resolution models."""
    
    print("🔥 FIRE Super-Resolution Demo")
    print("=" * 50)
    
    # Create test image
    hr_size = 64  # Use smaller size compatible with SwinIR
    scale = 4
    lr_size = hr_size // scale
    
    print(f"Creating test image ({hr_size}x{hr_size})...")
    hr_image = create_test_image(hr_size)
    lr_image = create_degraded_image(hr_image, scale)
    
    print(f"Degraded to low-resolution ({lr_size}x{lr_size})")
    
    # Models to test
    test_models = [
        ('Bicubic', None),  # Baseline
        ('EDSR Small', 'edsr_small'),
        ('Real-ESRGAN Medium', 'real_esrgan_medium'),
        ('EDSR Medium', 'edsr_medium')
    ]
    
    # Create figure
    fig, axes = plt.subplots(2, len(test_models), figsize=(16, 8))
    fig.suptitle('Super-Resolution Comparison (4x upscaling)', fontsize=16)
    
    results = {}
    
    for i, (name, model_name) in enumerate(test_models):
        print(f"\nTesting {name}...")
        
        if model_name is None:  # Bicubic baseline
            sr_image = F.interpolate(lr_image, scale_factor=scale, mode='bicubic', align_corners=False)
            params = 0
        else:
            # Create and run model
            if 'swinir' in model_name:
                model = create_model(model_name, upscale=scale)
            else:
                model = create_model(model_name, scale=scale)
            params = sum(p.numel() for p in model.parameters())
            
            model.eval()
            with torch.no_grad():
                sr_image = model(lr_image)
        
        # Store result
        results[name] = {
            'sr_image': sr_image,
            'parameters': params
        }
        
        # Plot LR image (top row)
        axes[0, i].imshow(lr_image[0].permute(1, 2, 0).clamp(0, 1))
        axes[0, i].set_title(f'LR Input\\n({lr_size}×{lr_size})')
        axes[0, i].axis('off')
        
        # Plot SR result (bottom row)
        axes[1, i].imshow(sr_image[0].permute(1, 2, 0).clamp(0, 1))
        title = f'{name}\\n({hr_size}×{hr_size})'
        if params > 0:
            if params >= 1e6:
                title += f'\\n{params/1e6:.1f}M params'
            else:
                title += f'\\n{params/1e3:.0f}K params'
        axes[1, i].set_title(title)
        axes[1, i].axis('off')
        
        print(f"  ✅ Output shape: {sr_image.shape}")
        if params > 0:
            print(f"  ✅ Parameters: {params:,}")
    
    plt.tight_layout()
    plt.savefig('super_resolution_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Demo results saved to 'super_resolution_demo.png'")
    
    # Print comparison
    print(f"\n🎯 COMPARISON SUMMARY:")
    print("-" * 50)
    for name, result in results.items():
        params = result['parameters']
        if params == 0:
            print(f"{name:<20}: Baseline method")
        else:
            print(f"{name:<20}: {params:,} parameters")
    
    return results


def main():
    """Main demo function."""
    results = demonstrate_models()
    
    print(f"\n💡 TIP: Try different models with:")
    print(f"   model = create_model('real_esrgan_large', scale=4)")
    print(f"   sr_image = model(lr_image)")
    
    print(f"\n🚀 Ready to enhance your images with FIRE!")


if __name__ == "__main__":
    main()