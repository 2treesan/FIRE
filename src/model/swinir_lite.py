# src/model/swinir_lite.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, window_size: int):
    """Split [B, H, W, C] into non-overlapping windows of size window_size."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """Merge windows back to [B, H, W, C]."""
    B = int(windows.shape[0] // (H // window_size * W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based MSA with relative position bias (minimal)."""
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # relative position bias
        ws = window_size
        num_relative = (2 * ws - 1) * (2 * ws - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative, num_heads))

        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        # support both torch>=1.10 (indexing='ij') and older
        try:
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        except TypeError:
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)  # [2, ws*ws]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 2]
        relative_coords[:, :, 0] += ws - 1
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 0] *= 2 * ws - 1
        relative_position_index = relative_coords.sum(-1)  # [N, N]
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x):  # x: [B*NW, N, C]
        Bn, N, C = x.shape
        qkv = self.qkv(x).reshape(Bn, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, Bn, heads, N, dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [Bn, heads, N, N]
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1
        )  # [N,N,heads]
        bias = bias.permute(2, 0, 1).unsqueeze(0)  # [1, heads, N, N]
        attn = (attn + bias).softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        x = self.proj(x)
        return x


class SwinBlock(nn.Module):
    """Minimal Swin block with optional shift; keeps spatial/channels constant."""
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4, shift: bool = False, mlp_ratio: float = 2.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):  # [B,C,H,W]
        B, C, H, W = x.shape
        shortcut = x                      # preserve residual at original size
        ws = self.window_size
        ss = self.shift_size

        # pad to multiples of window size
        pad_b = (ws - H % ws) % ws
        pad_r = (ws - W % ws) % ws
        x_pad = F.pad(x, (0, pad_r, 0, pad_b), mode='reflect')
        _, _, Hp, Wp = x_pad.shape

        # cyclic shift
        if ss > 0:
            x_pad = torch.roll(x_pad, shifts=(-ss, -ss), dims=(2, 3))

        # partition windows
        xw = x_pad.permute(0, 2, 3, 1).contiguous()               # [B,Hp,Wp,C]
        windows = window_partition(xw, ws).view(-1, ws * ws, C)   # [B*nW, N, C]

        # attention + FFN with pre-norm
        y = self.norm1(windows)
        y = self.attn(y)
        windows = windows + y

        y = self.norm2(windows)
        y = self.mlp(y)
        windows = windows + y

        # reverse windows
        xw = window_reverse(windows.view(-1, ws, ws, C), ws, Hp, Wp)  # [B,Hp,Wp,C]
        xw = xw.permute(0, 3, 1, 2).contiguous()                      # [B,C,Hp,Wp]

        # reverse shift
        if ss > 0:
            xw = torch.roll(xw, shifts=(ss, ss), dims=(2, 3))

        # crop back to original size and add residual
        x_out = xw[:, :, :H, :W]
        if x_out.size(2) != H or x_out.size(3) != W:  # extra guard (edge cases)
            x_out = x_out[:, :, :H, :W]
        return shortcut + x_out


class SwinIRLite(nn.Module):
    """Lightweight SwinIR-style super-resolution.
    I/O:
      Input  [B, C, H, W]   (LR)
      Output [B, C, H*scale, W*scale] in [0,1]
    """
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        embed_dim: int = 64,
        depths: int = 6,
        num_heads: int = 4,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        scale: int = 2,
        resi_scale: float = 1.0,
    ):
        super().__init__()
        self.scale = int(scale)
        self.shallow = nn.Conv2d(in_ch, embed_dim, 3, 1, 1)

        # trunk: alternating shifted windows
        blocks = [
            SwinBlock(embed_dim, window_size, num_heads, shift=(i % 2 == 1), mlp_ratio=mlp_ratio)
            for i in range(depths)
        ]
        self.trunk = nn.Sequential(*blocks)
        self.trunk_conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.resi_scale = resi_scale

        # upsampler
        self.upsampler = self._make_upsampler(self.scale, embed_dim, out_ch)

    @staticmethod
    def _make_upsampler(scale: int, nf: int, out_ch: int):
        m = []
        if scale in (2, 4, 8, 16):
            n_steps = int(round(math.log(scale, 2)))
            for _ in range(n_steps):
                m += [nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(True)]
        elif scale == 3:
            m += [nn.Conv2d(nf, nf * 9, 3, 1, 1), nn.PixelShuffle(3), nn.ReLU(True)]
        else:
            raise ValueError(f"Unsupported scale: {scale}. Use 2,3,4,8,16.")
        m += [nn.Conv2d(nf, out_ch, 3, 1, 1)]
        return nn.Sequential(*m)

    def forward(self, x):
        feat = self.shallow(x)
        trunk = self.trunk(feat)
        trunk = self.trunk_conv(trunk)
        feat = feat + trunk * self.resi_scale
        out = self.upsampler(feat)
        return out.clamp(0.0, 1.0)
