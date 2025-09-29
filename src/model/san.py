import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class _SecondOrderAttention(nn.Module):
    """
    Lightweight second-order channel attention.
    Computes channel-wise covariance (second-order statistics) over spatial dims,
    reduces it to a channel descriptor, then uses a small MLP (1x1 convs) to get
    per-channel weights.
    """

    def __init__(self, channels: int, reduction: int = 16, eps: float = 1e-6) -> None:
        super().__init__()
        r = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, r, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(r, channels, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        x_flat = x.view(b, c, n)
        mu = x_flat.mean(dim=2, keepdim=True)
        xc = x_flat - mu
        # covariance: CxC per batch
        cov = torch.bmm(xc, xc.transpose(1, 2)) / (max(1, n - 1))
        # reduce covariance to a per-channel descriptor (mean over rows)
        v = cov.mean(dim=2, keepdim=True)  # [B, C, 1]
        v = v.view(b, c, 1, 1)             # [B, C, 1, 1]
        w = self.mlp(v)                    # [B, C, 1, 1]
        return x * w


class _SAB(nn.Module):
    """Second-order Attention Block: conv -> relu -> conv -> SO-attn -> residual add"""

    def __init__(self, nf: int, reduction: int = 16, res_scale: float = 1.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
        )
        self.so = _SecondOrderAttention(nf, reduction)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        y = self.so(y)
        return x + self.res_scale * y


class _RG(nn.Module):
    """Residual Group for SAN: stack of SABs + conv, with group residual."""

    def __init__(self, nf: int, n_sab: int, reduction: int, res_scale: float) -> None:
        super().__init__()
        blocks: List[nn.Module] = [
            _SAB(nf, reduction=reduction, res_scale=res_scale) for _ in range(n_sab)
        ]
        blocks += [nn.Conv2d(nf, nf, 3, 1, 1)]
        self.body = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        return x + y


class _UpsamplerPixelShuffle(nn.Sequential):
    def __init__(self, scale: int, nf: int, out_ch: int):
        m: List[nn.Module] = []
        if scale in (2, 3):
            m += [nn.Conv2d(nf, nf * (scale ** 2), 3, 1, 1), nn.PixelShuffle(scale)]
        elif scale == 4:
            for _ in range(2):
                m += [nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2)]
        else:
            raise ValueError(f"Unsupported scale={scale}")
        m += [nn.Conv2d(nf, out_ch, 3, 1, 1)]
        super().__init__(*m)


class SAN(nn.Module):
    """Second-order Attention Network (SAN) - compact implementation.

    Kept small to be practical while retaining second-order attention:
    - Residual Groups composed of Second-order Attention Blocks
    - Global residual connection from shallow features to trunk output
    - PixelShuffle upsampler for 2x/3x/4x
    """

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        nf: int = 48,          # feature channels (small)
        n_rg: int = 6,         # number of residual groups (small)
        n_sab: int = 6,        # SAB per group (small)
        reduction: int = 8,    # reduction in second-order attention MLP
        res_scale: float = 1.0,
        scale: int = 2,
    ) -> None:
        super().__init__()
        if scale not in (2, 3, 4):
            raise ValueError(f"SAN only supports scale 2/3/4, got {scale}")

        self.in_ch, self.out_ch = in_ch, out_ch
        self.nf = nf
        self.n_rg, self.n_sab = n_rg, n_sab
        self.reduction = reduction
        self.res_scale = res_scale
        self.scale = int(scale)

        # Shallow feature extraction
        self.head = nn.Conv2d(in_ch, nf, 3, 1, 1)

        # Trunk: residual groups with second-order attention blocks
        groups: List[nn.Module] = [
            _RG(nf, n_sab=n_sab, reduction=reduction, res_scale=res_scale)
            for _ in range(n_rg)
        ]
        self.trunk = nn.Sequential(*groups)
        self.trunk_tail = nn.Conv2d(nf, nf, 3, 1, 1)

        # Upsampler
        self.upsampler = _UpsamplerPixelShuffle(self.scale, nf=nf, out_ch=out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != self.in_ch:
            raise ValueError(f"Input must be [B,{self.in_ch},h,w], got {tuple(x.shape)}")

        b, c, h, w = x.shape
        target_hw = (h * self.scale, w * self.scale)

        s = self.head(x)
        y = self.trunk_tail(self.trunk(s))
        y = y + s  # global residual
        y = self.upsampler(y)

        if (y.size(-2), y.size(-1)) != target_hw:
            y = F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)
        return y.clamp(0, 1)
