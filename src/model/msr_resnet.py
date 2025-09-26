import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN, như trong SRResNet/EDSR"""
    def __init__(self, nf=64):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x), inplace=True))

class Upsampler(nn.Sequential):
    """PixelShuffle-based upsampler"""
    def __init__(self, scale, nf, out_ch):
        m = []
        if scale in [2, 3]:
            m += [nn.Conv2d(nf, nf * (scale ** 2), 3, 1, 1),
                  nn.PixelShuffle(scale)]
        elif scale == 4:
            m += [nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2),
                  nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2)]
        else:
            raise ValueError(f"Unsupported scale {scale}")
        m += [nn.Conv2d(nf, out_ch, 3, 1, 1)]
        super().__init__(*m)

class MSRResNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, nf=64, nb=16, scale=2):
        super().__init__()
        self.head = nn.Conv2d(in_ch, nf, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualBlockNoBN(nf) for _ in range(nb)])
        self.tail = Upsampler(scale, nf, out_ch)

    def forward(self, x):
        fea = self.head(x)
        body_out = self.body(fea)
        out = self.tail(body_out + fea)
        return out.clamp(0, 1)
