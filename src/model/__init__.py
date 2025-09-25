"""
FIRE Model Registry

This module provides access to all available super-resolution models.
Models are organized by type and capability.
"""

from .edsr_lite import EDSRLite
from .srcnn import SRCNN, SRCNNPlusPlus
from .edsr import EDSR, EDSRLarge, EDSRBaseline
from .real_esrgan import RealESRGAN, RealESRGANLight

# Model registry for easy access
MODEL_REGISTRY = {
    # Restoration models (same resolution I/O)
    'edsr_lite': EDSRLite,
    
    # Classic SR models
    'srcnn': SRCNN,
    'srcnn_plus': SRCNNPlusPlus,
    
    # EDSR family (with upscaling)
    'edsr': EDSR,
    'edsr_large': EDSRLarge,
    'edsr_baseline': EDSRBaseline,
    
    # Real-world SR models
    'real_esrgan': RealESRGAN,
    'real_esrgan_light': RealESRGANLight,
}

def get_model(name: str):
    """Get model class by name."""
    if name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}' not found. Available models: {available}")
    return MODEL_REGISTRY[name]

def list_models():
    """List all available models."""
    return list(MODEL_REGISTRY.keys())

__all__ = [
    'EDSRLite', 'SRCNN', 'SRCNNPlusPlus', 'EDSR', 'EDSRLarge', 'EDSRBaseline',
    'RealESRGAN', 'RealESRGANLight', 'MODEL_REGISTRY', 'get_model', 'list_models'
]