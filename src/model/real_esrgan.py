"""
Real-ESRGAN model implementation.
Based on the paper: "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data"
https://arxiv.org/abs/2107.10833

This is a state-of-the-art model for real-world image super-resolution.
"""

import math
import functools
from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block used in RRDBNet."""
    
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with kaiming normal."""
        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance.
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""
    
    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance.
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block."""
    
    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 4, 
                 num_feat: int = 64, num_block: int = 23, num_grow_ch: int = 32):
        super().__init__()
        self.scale = scale
        
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        
        # Upsample
        if self.scale == 2:
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        elif self.scale == 4:
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        elif self.scale == 8:
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out.clamp(0, 1)


class UNetDiscriminatorSN(nn.Module):
    """U-Net discriminator with spectral normalization."""
    
    def __init__(self, num_in_ch: int = 3, num_feat: int = 64, skip_connection: bool = True):
        super().__init__()
        self.skip_connection = skip_connection
        norm = nn.utils.spectral_norm
        
        # The first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        
        # Down-sampling
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        
        # Up-sampling
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        
        # Extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x0 = self.lrelu(self.conv0(x))
        x1 = self.lrelu(self.conv1(x0))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))
        
        # Up-sampling
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = self.lrelu(self.conv4(x3))
        
        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = self.lrelu(self.conv5(x4))
        
        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = self.lrelu(self.conv6(x5))
        
        if self.skip_connection:
            x6 = x6 + x0
        
        # Extra convolutions
        out = self.lrelu(self.conv7(x6))
        out = self.lrelu(self.conv8(out))
        out = self.conv9(out)
        
        return out


class RealESRGAN(nn.Module):
    """
    Real-ESRGAN model for real-world super-resolution.
    
    This combines the RRDBNet generator with advanced training techniques
    for handling real-world degradations.
    """
    
    def __init__(self, scale: int = 4, num_in_ch: int = 3, num_out_ch: int = 3,
                 num_feat: int = 64, num_block: int = 23, num_grow_ch: int = 32):
        super().__init__()
        self.scale = scale
        self.generator = RRDBNet(num_in_ch, num_out_ch, scale, num_feat, num_block, num_grow_ch)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W] in range [0, 1]
            
        Returns:
            Super-resolved tensor [B, C, H*scale, W*scale] in range [0, 1]
        """
        return self.generator(x)


# Predefined model variants
class RealESRGANLarge(RealESRGAN):
    """Large Real-ESRGAN model for best quality."""
    def __init__(self, scale: int = 4, **kwargs):
        super().__init__(scale=scale, num_feat=64, num_block=23, num_grow_ch=32, **kwargs)


class RealESRGANMedium(RealESRGAN):
    """Medium Real-ESRGAN model balancing quality and speed."""
    def __init__(self, scale: int = 4, **kwargs):
        super().__init__(scale=scale, num_feat=48, num_block=16, num_grow_ch=24, **kwargs)


class RealESRGANSmall(RealESRGAN):
    """Small Real-ESRGAN model for fast inference."""
    def __init__(self, scale: int = 4, **kwargs):
        super().__init__(scale=scale, num_feat=32, num_block=8, num_grow_ch=16, **kwargs)