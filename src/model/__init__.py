"""
FIRE Model Zoo - State-of-the-art super-resolution models.

This module provides implementations of various state-of-the-art deep learning models
for image super-resolution.

Available Models:
- EDSRLite: Lightweight restoration model (same size input/output)
- EDSR: Enhanced Deep Super-Resolution with upscaling (2x, 3x, 4x, 8x)
- Real-ESRGAN: Real-world super-resolution with advanced degradation handling
- SwinIR: Transformer-based super-resolution model

Model Variants:
- Large: Best quality, highest parameters
- Medium: Good balance of quality and speed
- Small: Fast inference, lower parameters
"""

from typing import Dict, Type, Any
import torch.nn as nn

# Import existing model
from .edsr_lite import EDSRLite

# Import EDSR variants
from .edsr import EDSR, EDSRLarge, EDSRMedium, EDSRSmall

# Import Real-ESRGAN variants  
from .real_esrgan import (
    RealESRGAN, RealESRGANLarge, RealESRGANMedium, RealESRGANSmall,
    RRDBNet, UNetDiscriminatorSN
)

# Import SwinIR variants
from .swinir_simple import (
    SwinIRSimple, SwinIRSimpleLarge, SwinIRSimpleMedium, SwinIRSimpleSmall
)

# Model registry for easy access
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    # Restoration models (1x)
    'edsr_lite': EDSRLite,
    
    # EDSR Super-Resolution models
    'edsr': EDSR,
    'edsr_large': EDSRLarge,
    'edsr_medium': EDSRMedium,
    'edsr_small': EDSRSmall,
    
    # Real-ESRGAN models
    'real_esrgan': RealESRGAN,
    'real_esrgan_large': RealESRGANLarge,
    'real_esrgan_medium': RealESRGANMedium,
    'real_esrgan_small': RealESRGANSmall,
    'rrdb': RRDBNet,
    
    # SwinIR models
    'swinir': SwinIRSimple,
    'swinir_large': SwinIRSimpleLarge,
    'swinir_medium': SwinIRSimpleMedium,
    'swinir_small': SwinIRSimpleSmall,
}

# Model recommendations by use case
RECOMMENDED_MODELS = {
    'best_quality': 'edsr_large',
    'balanced': 'real_esrgan_medium', 
    'fastest': 'edsr_small',
    'transformer': 'swinir_medium',
    'real_world': 'real_esrgan_large',
    'lightweight': 'edsr_small',
}

# Model parameter counts (approximate)
MODEL_PARAMS = {
    'edsr_lite': 890_000,
    'edsr_large': 43_100_000,
    'edsr_medium': 6_100_000,
    'edsr_small': 930_000,
    'real_esrgan_large': 16_700_000,
    'real_esrgan_medium': 6_600_000,
    'real_esrgan_small': 1_500_000,
    'swinir_large': 10_200_000,
    'swinir_medium': 3_300_000,
    'swinir_small': 740_000,
}


def create_model(name: str, **kwargs) -> nn.Module:
    """
    Create a model by name.
    
    Args:
        name: Model name from MODEL_REGISTRY
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model name is not found
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{name}' not found. Available models: {available}")
    
    model_class = MODEL_REGISTRY[name]
    return model_class(**kwargs)


def list_models() -> Dict[str, Dict[str, Any]]:
    """
    List all available models with their information.
    
    Returns:
        Dictionary with model information
    """
    models_info = {}
    for name, model_class in MODEL_REGISTRY.items():
        models_info[name] = {
            'class': model_class.__name__,
            'params': MODEL_PARAMS.get(name, 'unknown'),
            'module': model_class.__module__,
            'doc': model_class.__doc__.strip() if model_class.__doc__ else None
        }
    return models_info


def get_recommended_model(use_case: str = 'balanced') -> str:
    """
    Get recommended model for a specific use case.
    
    Args:
        use_case: Use case type ('best_quality', 'balanced', 'fastest', etc.)
        
    Returns:
        Recommended model name
    """
    if use_case not in RECOMMENDED_MODELS:
        available = list(RECOMMENDED_MODELS.keys())
        raise ValueError(f"Use case '{use_case}' not found. Available: {available}")
    
    return RECOMMENDED_MODELS[use_case]


# Export all models and utilities
__all__ = [
    # Restoration models
    'EDSRLite',
    
    # EDSR models
    'EDSR', 'EDSRLarge', 'EDSRMedium', 'EDSRSmall',
    
    # Real-ESRGAN models
    'RealESRGAN', 'RealESRGANLarge', 'RealESRGANMedium', 'RealESRGANSmall',
    'RRDBNet', 'UNetDiscriminatorSN',
    
    # SwinIR models
    'SwinIRSimple', 'SwinIRSimpleLarge', 'SwinIRSimpleMedium', 'SwinIRSimpleSmall',
    
    # Utilities
    'create_model', 'list_models', 'get_recommended_model',
    'MODEL_REGISTRY', 'RECOMMENDED_MODELS', 'MODEL_PARAMS'
]