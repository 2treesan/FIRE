"""
SRCNN (Super-Resolution Convolutional Neural Network) implementation.
A classic and efficient model for single image super-resolution.

Paper: "Image Super-Resolution Using Deep Convolutional Networks"
Authors: Chao Dong, et al.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """
    SRCNN model for super-resolution.
    
    Architecture:
    - Patch extraction layer (9x9 conv)
    - Non-linear mapping layer (1x1 conv)  
    - Reconstruction layer (5x5 conv)
    
    Input:  [B, 3, H, W]  (bicubic upsampled LR image)
    Output: [B, 3, H, W]  (super-resolved image)
    """
    
    def __init__(self, 
                 in_ch: int = 3, 
                 out_ch: int = 3, 
                 n1: int = 64, 
                 n2: int = 32) -> None:
        """
        Initialize SRCNN model.
        
        Args:
            in_ch: Number of input channels (default: 3 for RGB)
            out_ch: Number of output channels (default: 3 for RGB)
            n1: Number of filters in patch extraction layer (default: 64)
            n2: Number of filters in non-linear mapping layer (default: 32)
        """
        super().__init__()
        
        # Patch extraction and representation
        self.conv1 = nn.Conv2d(in_ch, n1, kernel_size=9, padding=4)
        
        # Non-linear mapping
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=1, padding=0)
        
        # Reconstruction
        self.conv3 = nn.Conv2d(n2, out_ch, kernel_size=5, padding=2)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SRCNN.
        
        Args:
            x: Input tensor [B, 3, H, W] in range [0, 1]
            
        Returns:
            Super-resolved tensor [B, 3, H, W] in range [0, 1]
        """
        # Patch extraction and representation
        x = F.relu(self.conv1(x), inplace=True)
        
        # Non-linear mapping
        x = F.relu(self.conv2(x), inplace=True)
        
        # Reconstruction
        x = self.conv3(x)
        
        return x.clamp(0, 1)


class SRCNNPlusPlus(nn.Module):
    """
    Enhanced SRCNN with residual connection and deeper architecture.
    """
    
    def __init__(self, 
                 in_ch: int = 3, 
                 out_ch: int = 3, 
                 n1: int = 64, 
                 n2: int = 32, 
                 n3: int = 16,
                 use_residual: bool = True) -> None:
        """
        Initialize enhanced SRCNN model.
        
        Args:
            in_ch: Number of input channels
            out_ch: Number of output channels  
            n1: Filters in first layer
            n2: Filters in second layer
            n3: Filters in third layer
            use_residual: Whether to use residual connection
        """
        super().__init__()
        
        self.use_residual = use_residual
        
        # Patch extraction
        self.conv1 = nn.Conv2d(in_ch, n1, kernel_size=9, padding=4)
        
        # Non-linear mapping layers
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(n2, n3, kernel_size=3, padding=1)
        
        # Reconstruction
        self.conv4 = nn.Conv2d(n3, out_ch, kernel_size=5, padding=2)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection."""
        residual = x if self.use_residual else None
        
        # Patch extraction
        x = F.relu(self.conv1(x), inplace=True)
        
        # Non-linear mapping
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        
        # Reconstruction
        x = self.conv4(x)
        
        # Add residual connection if enabled
        if self.use_residual and residual is not None:
            x = x + residual
            
        return x.clamp(0, 1)