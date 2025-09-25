"""
Enhanced Deep Super-Resolution (EDSR) implementation.
A state-of-the-art model for single image super-resolution with upscaling.

Paper: "Enhanced Deep Residual Networks for Single Image Super-Resolution"
Authors: Bee Lim, et al.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    """Residual block without batch normalization."""
    
    def __init__(self, nf: int, res_scale: float = 0.1) -> None:
        """
        Initialize residual block.
        
        Args:
            nf: Number of features
            res_scale: Residual scaling factor
        """
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.res_scale = res_scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        res = x
        x = F.relu(self.conv1(x), inplace=True)
        x = self.conv2(x)
        return res + x * self.res_scale


class Upsampler(nn.Module):
    """Efficient sub-pixel convolution upsampler."""
    
    def __init__(self, scale: int, nf: int) -> None:
        """
        Initialize upsampler.
        
        Args:
            scale: Upscaling factor (2, 3, or 4)
            nf: Number of input features
        """
        super().__init__()
        
        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules.append(nn.Conv2d(nf, 4 * nf, 3, 1, 1))
                modules.append(nn.PixelShuffle(2))
        elif scale == 3:
            modules.append(nn.Conv2d(nf, 9 * nf, 3, 1, 1))
            modules.append(nn.PixelShuffle(3))
        else:
            raise NotImplementedError(f'Scale {scale} is not supported')
        
        self.body = nn.Sequential(*modules)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through upsampler."""
        return self.body(x)


class EDSR(nn.Module):
    """
    Enhanced Deep Super-Resolution Network.
    
    Input:  [B, 3, H, W]  (Low-resolution image)
    Output: [B, 3, scale*H, scale*W]  (Super-resolved image)
    """
    
    def __init__(self, 
                 scale: int = 2,
                 in_ch: int = 3, 
                 out_ch: int = 3, 
                 nf: int = 64, 
                 nb: int = 16, 
                 res_scale: float = 0.1) -> None:
        """
        Initialize EDSR model.
        
        Args:
            scale: Upscaling factor (2, 3, or 4)
            in_ch: Number of input channels
            out_ch: Number of output channels
            nf: Number of features
            nb: Number of residual blocks
            res_scale: Residual scaling factor
        """
        super().__init__()
        
        self.scale = scale
        
        # Head: initial feature extraction
        self.head = nn.Conv2d(in_ch, nf, 3, 1, 1)
        
        # Body: residual blocks
        body = [ResidualBlock(nf, res_scale) for _ in range(nb)]
        body.append(nn.Conv2d(nf, nf, 3, 1, 1))
        self.body = nn.Sequential(*body)
        
        # Tail: upsampling and output
        if scale > 1:
            self.upsampler = Upsampler(scale, nf)
            self.tail = nn.Conv2d(nf, out_ch, 3, 1, 1)
        else:
            self.upsampler = None
            self.tail = nn.Conv2d(nf, out_ch, 3, 1, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of EDSR.
        
        Args:
            x: Input LR tensor [B, 3, H, W] in range [0, 1]
            
        Returns:
            Super-resolved tensor [B, 3, scale*H, scale*W] in range [0, 1]
        """
        # Head
        x = self.head(x)
        
        # Body with global residual connection
        res = x
        x = self.body(x)
        x = x + res
        
        # Tail: upsampling and output
        if self.upsampler is not None:
            x = self.upsampler(x)
        x = self.tail(x)
        
        return x.clamp(0, 1)


class EDSRLarge(nn.Module):
    """
    Large version of EDSR with more parameters for better performance.
    """
    
    def __init__(self, 
                 scale: int = 2,
                 in_ch: int = 3, 
                 out_ch: int = 3, 
                 nf: int = 256, 
                 nb: int = 32, 
                 res_scale: float = 0.1) -> None:
        """Initialize large EDSR model with more parameters."""
        super().__init__()
        
        self.scale = scale
        
        # Head
        self.head = nn.Conv2d(in_ch, nf, 3, 1, 1)
        
        # Body
        body = [ResidualBlock(nf, res_scale) for _ in range(nb)]
        body.append(nn.Conv2d(nf, nf, 3, 1, 1))
        self.body = nn.Sequential(*body)
        
        # Tail
        if scale > 1:
            self.upsampler = Upsampler(scale, nf)
            self.tail = nn.Conv2d(nf, out_ch, 3, 1, 1)
        else:
            self.upsampler = None
            self.tail = nn.Conv2d(nf, out_ch, 3, 1, 1)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of large EDSR."""
        # Head
        x = self.head(x)
        
        # Body with global residual connection
        res = x
        x = self.body(x)
        x = x + res
        
        # Tail
        if self.upsampler is not None:
            x = self.upsampler(x)
        x = self.tail(x)
        
        return x.clamp(0, 1)


class EDSRBaseline(nn.Module):
    """
    Baseline EDSR model with batch normalization for comparison.
    """
    
    def __init__(self, 
                 scale: int = 2,
                 in_ch: int = 3, 
                 out_ch: int = 3, 
                 nf: int = 64, 
                 nb: int = 16) -> None:
        """Initialize baseline EDSR with batch normalization."""
        super().__init__()
        
        self.scale = scale
        
        # Head
        self.head = nn.Conv2d(in_ch, nf, 3, 1, 1)
        
        # Body with batch normalization
        body = []
        for _ in range(nb):
            body.extend([
                nn.Conv2d(nf, nf, 3, 1, 1),
                nn.BatchNorm2d(nf),
                nn.ReLU(inplace=True),
                nn.Conv2d(nf, nf, 3, 1, 1),
                nn.BatchNorm2d(nf),
            ])
        body.append(nn.Conv2d(nf, nf, 3, 1, 1))
        self.body = nn.Sequential(*body)
        
        # Tail
        if scale > 1:
            self.upsampler = Upsampler(scale, nf)
            self.tail = nn.Conv2d(nf, out_ch, 3, 1, 1)
        else:
            self.upsampler = None
            self.tail = nn.Conv2d(nf, out_ch, 3, 1, 1)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of baseline EDSR."""
        # Head
        x = self.head(x)
        
        # Body with global residual connection
        res = x
        x = self.body(x)
        x = x + res
        
        # Tail
        if self.upsampler is not None:
            x = self.upsampler(x)
        x = self.tail(x)
        
        return x.clamp(0, 1)