"""
Enhanced Deep Super-Resolution (EDSR) model implementation.
Based on the paper: "Enhanced Deep Residual Networks for Single Image Super-Resolution"
https://arxiv.org/abs/1707.02921

This is a state-of-the-art super-resolution model with proper upscaling capabilities.
"""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanShift(nn.Conv2d):
    """Mean shift layer for data normalization."""
    def __init__(self, rgb_range: int = 1, rgb_mean: Tuple[float, float, float] = (0.4488, 0.4371, 0.4040), 
                 rgb_std: Tuple[float, float, float] = (1.0, 1.0, 1.0), sign: int = -1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    """Residual block with improved architecture."""
    def __init__(self, n_feats: int, kernel_size: int = 3, bias: bool = True, 
                 bn: bool = False, act: str = 'relu', res_scale: float = 1):
        super().__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                if act.lower() == 'relu':
                    m.append(nn.ReLU(inplace=True))
                elif act.lower() == 'lrelu':
                    m.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    """Efficient sub-pixel upsampling layer."""
    def __init__(self, conv, scale: int, n_feats: int, bn: bool = False, act: bool = False, bias: bool = True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(inplace=True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(inplace=True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super().__init__(*m)


def default_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool = True):
    """Default convolution layer."""
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)


class EDSR(nn.Module):
    """
    Enhanced Deep Super-Resolution model with proper upscaling.
    
    Args:
        n_resblocks: Number of residual blocks
        n_feats: Number of feature channels
        res_scale: Residual scaling factor
        scale: Super-resolution scale factor (2, 3, 4, 8)
        n_colors: Number of input/output channels (typically 3 for RGB)
        conv: Convolution function to use
    """
    
    def __init__(self, n_resblocks: int = 32, n_feats: int = 256, res_scale: float = 0.1,
                 scale: int = 4, n_colors: int = 3, conv=default_conv):
        super().__init__()
        
        self.scale = scale
        self.n_colors = n_colors
        
        # RGB mean for normalization (ImageNet stats)
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(1, rgb_mean, rgb_std)
        
        # Define head module
        m_head = [conv(n_colors, n_feats, 3)]
        
        # Define body module  
        m_body = [
            ResBlock(n_feats, 3, res_scale=res_scale, act='relu') 
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, 3))
        
        # Define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, 3)
        ]
        
        self.add_mean = MeanShift(1, rgb_mean, rgb_std, 1)
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W] in range [0, 1]
            
        Returns:
            Super-resolved tensor [B, C, H*scale, W*scale] in range [0, 1]
        """
        # Normalize input
        x = self.sub_mean(x)
        
        # Extract features
        x = self.head(x)
        
        # Process through residual blocks
        res = self.body(x)
        res += x  # Global residual connection
        
        # Upsampling and final convolution
        x = self.tail(res)
        
        # Denormalize output
        x = self.add_mean(x)
        
        return x.clamp(0, 1)


class EDSRLarge(EDSR):
    """Large EDSR model for best quality."""
    def __init__(self, scale: int = 4, **kwargs):
        super().__init__(n_resblocks=32, n_feats=256, res_scale=0.1, scale=scale, **kwargs)


class EDSRMedium(EDSR):
    """Medium EDSR model balancing quality and speed."""
    def __init__(self, scale: int = 4, **kwargs):
        super().__init__(n_resblocks=16, n_feats=128, res_scale=0.2, scale=scale, **kwargs)


class EDSRSmall(EDSR):
    """Small EDSR model for fast inference."""
    def __init__(self, scale: int = 4, **kwargs):
        super().__init__(n_resblocks=8, n_feats=64, res_scale=0.2, scale=scale, **kwargs)