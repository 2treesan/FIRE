
import torch
import torch.nn as nn
import torch.nn.functional as F

class CALayer(nn.Module):
    """Channel Attention (SE) block used in RCAN.
    y = x * sigmoid(Conv(ReLU(Conv(GAPool(x)))))
    """
    def __init__(self, nf: int, reduction: int = 16) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(nf, nf // reduction, 1, 1, 0)
        self.conv2 = nn.Conv2d(nf // reduction, nf, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        s = F.relu(self.conv1(s), inplace=True)
        s = torch.sigmoid(self.conv2(s))
        return x * s


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB)."""
    def __init__(self, nf: int, res_scale: float = 1.0, reduction: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
        )
        self.ca = CALayer(nf, reduction)
        self.res_scale = res_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        y = self.ca(y)
        return x + y * self.res_scale


class ResidualGroup(nn.Module):
    """A group of RCABs followed by a conv, with a long skip connection."""
    def __init__(self, nf: int, nb: int, res_scale: float = 1.0, reduction: int = 16) -> None:
        super().__init__()
        blocks = [RCAB(nf, res_scale=res_scale, reduction=reduction) for _ in range(nb)]
        blocks.append(nn.Conv2d(nf, nf, 3, 1, 1))
        self.body = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class Upsampler(nn.Sequential):
    """PixelShuffle-based upsampler that supports x2/x3/x4 scales."""
    def __init__(self, scale: int, nf: int, out_ch: int = 3) -> None:
        m = []
        if scale in (2, 4, 8, 16):
            n_steps = int(round(torch.log2(torch.tensor(scale)).item()))
            for _ in range(n_steps):
                m += [nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(inplace=True)]
        elif scale == 3:
            m += [nn.Conv2d(nf, nf * 9, 3, 1, 1), nn.PixelShuffle(3), nn.ReLU(inplace=True)]
        else:
            raise ValueError(f"Unsupported scale: {scale}. Use 2,3,4,8,16.")
        m += [nn.Conv2d(nf, out_ch, 3, 1, 1)]
        super().__init__(*m)


class RCAN(nn.Module):
    """RCAN: Residual Channel Attention Network (PSNR-oriented SOTA for SR).

    This implementation keeps the FIRE I/O contract:
      - Input:  LR image tensor of shape [B, C, H, W], with C=in_ch
      - Output: SR image tensor of shape [B, C, H*scale, W*scale], values in [0,1]

    Args:
        in_ch:    number of input channels (3 for RGB)
        out_ch:   number of output channels (3 for RGB)
        nf:       base number of feature maps
        nb:       number of RCAB blocks per residual group
        ng:       number of residual groups
        scale:    upsampling scale factor (2/3/4/... as supported by Upsampler)
        res_scale: residual scaling inside RCABs (typ. 0.1~1.0)
        reduction: channel reduction ratio for the CA layer (typ. 16)
    """
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        nf: int = 64,
        nb: int = 10,
        ng: int = 10,
        scale: int = 2,
        res_scale: float = 1.0,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.scale = int(scale)

        # Head
        self.head = nn.Conv2d(in_ch, nf, 3, 1, 1)

        # Body: ng residual groups, each with nb RCABs
        groups = [ResidualGroup(nf, nb, res_scale=res_scale, reduction=reduction) for _ in range(ng)]
        self.body = nn.Sequential(*groups, nn.Conv2d(nf, nf, 3, 1, 1))

        # Tail: upsampler -> last conv
        self.tail = Upsampler(self.scale, nf, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.head(x)                      # [B,nf,H,W]
        b = self.body(s)                      # [B,nf,H,W]
        feat = s + b                          # long skip

        # Upsample to HR
        y = self.tail(feat)                   # [B,out_ch,H*scale,W*scale]

        return y.clamp(0.0, 1.0)
