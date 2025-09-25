"""
Real-ESRGAN inspired model for practical super-resolution.
Designed to handle real-world degraded images effectively.

Based on: "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data"
Authors: Xintao Wang, et al.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB) with dense connections."""
    
    def __init__(self, nf: int = 32, gc: int = 32) -> None:
        """
        Initialize RDB.
        
        Args:
            nf: Number of input features
            gc: Growth channel (number of output features for each conv layer)
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dense connections."""
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        
        # Residual scaling for stability
        return x5 * 0.2 + x


class ResidualInResidualDenseBlock(nn.Module):
    """Residual-in-Residual Dense Block (RRDB)."""
    
    def __init__(self, nf: int = 32, gc: int = 32) -> None:
        """Initialize RRDB with nested residual connections."""
        super().__init__()
        
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through nested RDBs."""
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        
        # Residual scaling for stability
        return out * 0.2 + x


class RealESRGAN(nn.Module):
    """
    Real-ESRGAN model for practical super-resolution.
    
    Input:  [B, 3, H, W]  (Low-resolution image)
    Output: [B, 3, scale*H, scale*W]  (Super-resolved image)
    """
    
    def __init__(self, 
                 scale: int = 4,
                 in_ch: int = 3, 
                 out_ch: int = 3, 
                 nf: int = 64, 
                 nb: int = 23, 
                 gc: int = 32) -> None:
        """
        Initialize Real-ESRGAN model.
        
        Args:
            scale: Upscaling factor (2, 4, or 8)
            in_ch: Number of input channels
            out_ch: Number of output channels
            nf: Number of features
            nb: Number of RRDB blocks
            gc: Growth channel in RDB
        """
        super().__init__()
        
        self.scale = scale
        
        # Initial convolution
        self.conv_first = nn.Conv2d(in_ch, nf, 3, 1, 1)
        
        # RRDB blocks
        self.body = nn.Sequential(*[
            ResidualInResidualDenseBlock(nf, gc) for _ in range(nb)
        ])
        
        # Body convolution
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        
        # Upsampling layers
        if scale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
            self.upconv2 = None
            self.pixel_shuffle2 = None
        elif scale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.pixel_shuffle2 = nn.PixelShuffle(2)
        elif scale == 8:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.pixel_shuffle2 = nn.PixelShuffle(2)
            self.upconv3 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.pixel_shuffle3 = nn.PixelShuffle(2)
        else:
            raise NotImplementedError(f'Scale {scale} is not supported')
        
        # High-resolution convolutions
        self.hrconv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_ch, 3, 1, 1)
        
        # Activation
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Real-ESRGAN.
        
        Args:
            x: Input LR tensor [B, 3, H, W] in range [0, 1]
            
        Returns:
            Super-resolved tensor [B, 3, scale*H, scale*W] in range [0, 1]
        """
        # Initial feature extraction
        feat = self.conv_first(x)
        
        # RRDB blocks with global residual connection
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        
        # Upsampling
        feat = self.lrelu(self.pixel_shuffle1(self.upconv1(feat)))
        
        if self.scale >= 4:
            feat = self.lrelu(self.pixel_shuffle2(self.upconv2(feat)))
        
        if self.scale == 8:
            feat = self.lrelu(self.pixel_shuffle3(self.upconv3(feat)))
        
        # High-resolution processing
        feat = self.lrelu(self.hrconv(feat))
        out = self.conv_last(feat)
        
        return out.clamp(0, 1)


class RealESRGANLight(nn.Module):
    """
    Lightweight version of Real-ESRGAN for faster inference.
    """
    
    def __init__(self, 
                 scale: int = 4,
                 in_ch: int = 3, 
                 out_ch: int = 3, 
                 nf: int = 32, 
                 nb: int = 12, 
                 gc: int = 16) -> None:
        """Initialize lightweight Real-ESRGAN."""
        super().__init__()
        
        self.scale = scale
        
        # Initial convolution
        self.conv_first = nn.Conv2d(in_ch, nf, 3, 1, 1)
        
        # Reduced RRDB blocks
        self.body = nn.Sequential(*[
            ResidualInResidualDenseBlock(nf, gc) for _ in range(nb)
        ])
        
        # Body convolution
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        
        # Simplified upsampling
        upsample_layers = []
        if scale == 2:
            upsample_layers.extend([
                nn.Conv2d(nf, nf * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        elif scale == 4:
            upsample_layers.extend([
                nn.Conv2d(nf, nf * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(nf, nf * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ])
        
        self.upsampling = nn.Sequential(*upsample_layers)
        
        # Output layers
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of lightweight Real-ESRGAN."""
        # Feature extraction
        feat = self.conv_first(x)
        
        # Body with global residual
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        
        # Upsampling
        feat = self.upsampling(feat)
        
        # High-resolution processing
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)
        
        return out.clamp(0, 1)