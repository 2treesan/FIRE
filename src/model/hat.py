import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# -------------------- small helper upsampler (ICNR) --------------------
def _icnr_(w: torch.Tensor, scale: int, init=nn.init.kaiming_normal_):
    out_channels, in_channels, kh, kw = w.shape
    if out_channels % (scale ** 2) != 0:
        return w
    new_shape = (out_channels // (scale ** 2), in_channels, kh, kw)
    subkernel = torch.zeros(new_shape, device=w.device, dtype=w.dtype)
    init(subkernel)
    subkernel = subkernel.repeat_interleave(scale ** 2, dim=0)
    with torch.no_grad():
        w.copy_(subkernel)
    return w

class _UpsamplerPixelShuffleICNR(nn.Module):
    def __init__(self, scale: int, nf: int, out_ch: int):
        super().__init__()
        self.scale = scale
        if scale in (2, 3):
            self.conv = nn.Conv2d(nf, nf * (scale ** 2), 3, 1, 1)
            self.ps = nn.PixelShuffle(scale)
            self.tail = nn.Conv2d(nf, out_ch, 3, 1, 1)
            _icnr_(self.conv.weight, scale=scale)
        elif scale == 4:
            self.conv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.ps1 = nn.PixelShuffle(2)
            self.conv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.ps2 = nn.PixelShuffle(2)
            self.tail = nn.Conv2d(nf, out_ch, 3, 1, 1)
            _icnr_(self.conv1.weight, scale=2)
            _icnr_(self.conv2.weight, scale=2)
        else:
            raise ValueError(f"Unsupported scale={scale}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale in (2, 3):
            x = self.ps(self.conv(x))
            return self.tail(x)
        else:
            x = self.ps1(self.conv1(x))
            x = self.ps2(self.conv2(x))
            return self.tail(x)

# -------------------- window utils --------------------
def window_partition(x: torch.Tensor, window_size: int):
    """
    x: (B, C, H, W)
    return: (num_windows*B, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    ws = window_size
    pad_h = (ws - H % ws) % ws
    pad_w = (ws - W % ws) % ws
    if pad_h != 0 or pad_w != 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))  # pad right and bottom
        H = H + pad_h
        W = W + pad_w
    x = x.view(B, C, H // ws, ws, W // ws, ws)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # B, H/ws, W/ws, ws, ws, C
    windows = x.view(-1, ws, ws, C)  # (num_windows*B, ws, ws, C)
    return windows, H, W, pad_h, pad_w

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int, pad_h: int, pad_w: int, B: int):
    """
    windows: (num_windows*B, ws, ws, C)
    returns: (B, C, H-pad_h, W-pad_w)
    """
    ws = window_size
    C = windows.shape[-1]
    x = windows.view(B, H // ws, W // ws, ws, ws, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # B, C, H//ws, ws, W//ws, ws
    x = x.view(B, C, H, W)
    if pad_h != 0 or pad_w != 0:
        x = x[:, :, : H - pad_h, : W - pad_w]
    return x

# -------------------- HAT block (simplified lite) --------------------
class HATBlock(nn.Module):
    """
    Lightweight HAT-style block with:
      - window-based multi-head self-attention (W-MSA)
      - feed-forward (MLP)
      - residuals & LayerNorm (pre-norm)
    """
    def __init__(self, dim: int, num_heads: int = 2, window_size: int = 8, ffn_expansion: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.ffn_hidden = int(dim * ffn_expansion)

        # LayerNorm operates on last dim of token representation
        self.norm1 = nn.LayerNorm(dim)
        # project to qkv via a linear; we'll use nn.MultiheadAttention which expects (L, N, E) or batch_first
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, self.ffn_hidden),
            nn.GELU(),
            nn.Linear(self.ffn_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        ws = self.window_size

        # partition windows
        windows, Hp, Wp, pad_h, pad_w = window_partition(x, ws)  # (num_w*B, ws, ws, C)
        nw = windows.shape[0]
        # flatten spatial within window => tokens (nw, ws*ws, C)
        tokens = windows.view(nw, ws * ws, C)  # batch= nw

        # PreNorm
        tokens_norm = self.norm1(tokens)  # LayerNorm over last dim

        # MHA expects (batch, seq_len, emb)
        attn_out, _ = self.attn(tokens_norm, tokens_norm, tokens_norm, need_weights=False)
        attn_out = self.proj(attn_out)
        tokens = tokens + attn_out  # residual

        # FFN
        tokens_norm2 = self.norm2(tokens)
        mlp_out = self.mlp(tokens_norm2)
        tokens = tokens + mlp_out  # residual

        # reshape back
        windows = tokens.view(nw, ws, ws, C)
        x = window_reverse(windows, ws, Hp, Wp, pad_h, pad_w, B)
        return x

# -------------------- main HAT model (EDSR style integration) --------------------
class HAT(nn.Module):
    """
    HAT integrated into an EDSR-style pipeline.
    - head: conv in_ch -> nf
    - body: nb x HATBlock (operating on NF channels in LR space)
    - merge conv + residual add
    - tail: upsampler (pixelshuffle_icnr) to HR
    Interface matches your EDSR: accepts (in_ch,out_ch,nf,nb,scale,mode,method) but
    only 'post_upsample' + 'pixelshuffle_icnr' intended (others minimal support).
    """
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        nf: int = 48,
        nb: int = 6,
        res_scale: float = 0.1,   # not heavily used but kept for API
        scale: int = 2,
        num_heads: int = 2,
        window_size: int = 8,
        ffn_expansion: float = 2.0,
        mode: str = "post_upsample",
        method: str = "pixelshuffle_icnr",
    ) -> None:
        super().__init__()
        # only basic validation (keep API simple)
        if mode != "post_upsample":
            raise ValueError("HAT currently supports mode='post_upsample' only for tiny CPU variant.")
        if method not in ("pixelshuffle", "pixelshuffle_icnr", "deconvolution"):
            raise ValueError("HAT tail method must be 'pixelshuffle' or 'pixelshuffle_icnr' or 'deconvolution'")

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nf = nf
        self.nb = int(nb)
        self.scale = int(scale)
        self.res_scale = res_scale
        self.num_heads = num_heads
        self.window_size = int(window_size)
        self.ffn_expansion = ffn_expansion

        # head conv: map input RGB -> nf channels (LR space)
        self.head = nn.Conv2d(in_ch, nf, kernel_size=3, stride=1, padding=1)

        # body: sequence of HAT blocks (operate on nf channels)
        blocks = []
        for _ in range(self.nb):
            blocks.append(HATBlock(dim=nf, num_heads=self.num_heads, window_size=self.window_size, ffn_expansion=self.ffn_expansion))
        self.body = nn.Sequential(*blocks)

        # merge
        self.merge = nn.Conv2d(nf, nf, 3, 1, 1)

        # tail / upsampler
        if method == "pixelshuffle_icnr":
            self.tail = _UpsamplerPixelShuffleICNR(self.scale, nf, out_ch)
        elif method == "pixelshuffle":
            # fallback: basic pixelshuffle upsampler
            self.tail = nn.Sequential(
                nn.Conv2d(nf, nf * (self.scale ** 2), 3, 1, 1),
                nn.PixelShuffle(self.scale),
                nn.Conv2d(nf, out_ch, 3, 1, 1)
            )
            # init conv icnr-like for pixelshuffle to reduce checkerboard
            _icnr_(self.tail[0].weight, scale=self.scale)
        else:
            # deconv fallback
            self.tail = nn.ConvTranspose2d(nf, out_ch, kernel_size=4, stride=self.scale, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != self.in_ch:
            raise ValueError(f"Input must be [B,{self.in_ch},h,w], got {tuple(x.shape)}")

        b, c, h, w = x.shape
        target_hw = (h * self.scale, w * self.scale)

        s = self.head(x)           # [B, nf, h, w]
        res = s
        y = self.body(s)           # pass through HAT blocks
        y = self.merge(y) + res    # residual connection

        y = self.tail(y)           # upsample
        # ensure exact size
        if (y.size(-2), y.size(-1)) != target_hw:
            y = F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)
        return y.clamp(0, 1)
