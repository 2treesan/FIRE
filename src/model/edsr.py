import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict, Tuple, List

# -------------------- core blocks --------------------
class _ResBlock(nn.Module):
    def __init__(self, nf: int, res_scale: float = 0.1) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.c2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.res_scale = res_scale
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res_scale * (self.c2(F.relu(self.c1(x), inplace=True)))

class _UpsamplerPixelShuffle(nn.Sequential):
    """PixelShuffle upsampler cho scale 2/3/4 (chuẩn EDSR)."""
    def __init__(self, scale: int, nf: int, out_ch: int):
        m = []
        if scale in (2, 3):
            m += [nn.Conv2d(nf, nf * (scale ** 2), 3, 1, 1), nn.PixelShuffle(scale)]
        elif scale == 4:
            for _ in range(2):
                m += [nn.Conv2d(nf, nf * 4, 3, 1, 1), nn.PixelShuffle(2)]
        else:
            raise ValueError(f"Unsupported scale={scale}")
        m += [nn.Conv2d(nf, out_ch, 3, 1, 1)]
        super().__init__(*m)

class _UpsamplerDeconv(nn.Sequential):
    """Transposed Convolution upsampler cho scale 2/3/4."""
    def __init__(self, scale: int, nf: int, out_ch: int):
        m = []
        if scale == 2:
            m += [nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1)]
        elif scale == 3:
            m += [nn.ConvTranspose2d(nf, nf, kernel_size=7, stride=3, padding=2)]
        elif scale == 4:
            m += [nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1),
                  nn.ReLU(inplace=True),
                  nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1)]
        else:
            raise ValueError(f"Unsupported scale={scale}")
        m += [nn.Conv2d(nf, out_ch, 3, 1, 1)]
        super().__init__(*m)

# -------------------- advanced upsamplers --------------------
class _CARAFE2d(nn.Module):
    """
    Content-Aware ReAssembly of FEatures (giản lược).
    - Dự đoán kernel (k*k*s*s) theo nội dung, softmax theo k*k.
    - Trộn lại các patch unfold để tạo [B,C,H*s,W*s].
    Lưu ý: dùng shared kernel cho mọi channel (chuẩn CARAFE).
    """
    def __init__(self, channels: int, scale: int = 2, k: int = 5, m: int = 64):
        super().__init__()
        self.channels, self.scale, self.k = channels, scale, k
        self.comp = nn.Conv2d(channels, m, 1, 1, 0)
        self.kernel_pred = nn.Conv2d(m, (k * k) * (scale * scale), 1, 1, 0)
        self._eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s, k = self.scale, self.k

        # predict kernels
        w = self.kernel_pred(F.relu(self.comp(x), inplace=True))  # [B, k*k*s*s, H, W]
        w = F.pixel_shuffle(w, upscale_factor=s)                  # [B, k*k, H*s, W*s]
        w = w.view(B, k * k, H * s, W * s)
        w = F.softmax(w, dim=1)                                   # normalize over k*k

        # unfold LR features
        pad = k // 2
        unfold = F.unfold(x, kernel_size=k, padding=pad)          # [B, C*k*k, H*W]
        unfold = unfold.view(B, C, k * k, H, W)

        # upsample index map
        # gather per (Hs,Ws) from nearest LR center
        # (giản lược, mapping nearest-center; đủ ổn trong thực nghiệm nhanh)
        grid_y = torch.arange(H * s, device=x.device) // s
        grid_x = torch.arange(W * s, device=x.device) // s
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")    # [Hs, Ws]
        gather = unfold[:, :, :, gy, gx]                          # [B,C,k*k,Hs,Ws]

        # weighted sum over k*k
        out = (gather * w.unsqueeze(1)).sum(dim=2)                # [B,C,Hs,Ws]
        return out

def _icnr_(w: torch.Tensor, scale: int, init=nn.init.kaiming_normal_):
    """
    ICNR init cho PixelShuffle để giảm checkerboard (Aitken et al.).
    """
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
    """
    PixelShuffle + ICNR init (giảm checkerboard) cho scale 2/3/4.
    """
    def __init__(self, scale: int, nf: int, out_ch: int):
        super().__init__()
        self.scale = scale
        if scale in (2, 3):
            self.conv = nn.Conv2d(nf, nf * (scale ** 2), 3, 1, 1)
            self.ps = nn.PixelShuffle(scale)
            self.tail = nn.Conv2d(nf, out_ch, 3, 1, 1)
        elif scale == 4:
            self.conv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.ps1 = nn.PixelShuffle(2)
            self.conv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1)
            self.ps2 = nn.PixelShuffle(2)
            self.tail = nn.Conv2d(nf, out_ch, 3, 1, 1)
        else:
            raise ValueError(f"Unsupported scale={scale}")

        # ICNR init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m is not self.tail:
                _icnr_(m.weight, scale=2 if scale == 4 else scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale in (2, 3):
            x = self.ps(self.conv(x))
            return self.tail(x)
        else:
            x = self.ps1(self.conv1(x))
            x = self.ps2(self.conv2(x))
            return self.tail(x)

class _MetaUp2d(nn.Module):
    """
    Meta-Upscale (giản lược): dự đoán kernel động (k*k) theo nội dung cho từng vị trí HR,
    áp dụng riêng cho mỗi channel nf (shared kernel across channels để tiết kiệm).
    Thực tế gần với CARAFE nhưng cho phép thêm một tầng trộn kênh sau upsample.
    """
    def __init__(self, channels: int, scale: int = 2, k: int = 5, m: int = 64, out_ch: int = None):
        super().__init__()
        self.carafe = _CARAFE2d(channels, scale=scale, k=k, m=m)
        self.mix = nn.Identity() if (out_ch is None or out_ch == channels) else nn.Conv2d(channels, out_ch, 1, 1, 0)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(self.carafe(x))

# -------------------- pre strategies (modules) --------------------
class _IdentityPre(nn.Module):
    def forward(self, x: torch.Tensor, target_hw: Tuple[int,int]) -> torch.Tensor:
        return x

class _BicubicPre(nn.Module):
    def forward(self, x: torch.Tensor, target_hw: Tuple[int,int]) -> torch.Tensor:
        H, W = target_hw
        return F.interpolate(x, size=(H, W), mode="bicubic", align_corners=False)

class _LearnablePre(nn.Module):
    """RGB -> nf*(s^2) -> PixelShuffle(s) => [B,nf,H,W]"""
    def __init__(self, in_ch: int, nf: int, scale: int) -> None:
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, nf * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale),
        )
    def forward(self, x: torch.Tensor, target_hw: Tuple[int,int]) -> torch.Tensor:
        return self.pre(x)

class _CARAFEPre(nn.Module):
    """Upsample trong không gian ảnh bằng CARAFE (content-aware). Output [B,C,H,W]."""
    def __init__(self, in_ch: int, scale: int, k: int = 5, m: int = 64) -> None:
        super().__init__()
        self.up = _CARAFE2d(in_ch, scale=scale, k=k, m=m)
    def forward(self, x: torch.Tensor, target_hw: Tuple[int,int]) -> torch.Tensor:
        return self.up(x)

# -------------------- registries --------------------
SUPPORTED: Dict[str, List[str]] = {
    "pre_upsample":  ["learnable", "bicubic", "carafe"],
    "post_upsample": ["deconvolution", "pixelshuffle", "pixelshuffle_icnr", "meta"],
}

# (mode, method) -> factory returning (pre_module, head_module)
HEAD_BUILDERS: Dict[Tuple[str, str], Callable[[int,int,int], Tuple[nn.Module, nn.Module]]] = {
    ("post_upsample", "pixelshuffle"):      lambda in_ch, nf, scale: (_IdentityPre(), nn.Conv2d(in_ch, nf, 3, 1, 1)),
    ("post_upsample", "deconvolution"):     lambda in_ch, nf, scale: (_IdentityPre(), nn.Conv2d(in_ch, nf, 3, 1, 1)),
    ("post_upsample", "pixelshuffle_icnr"): lambda in_ch, nf, scale: (_IdentityPre(), nn.Conv2d(in_ch, nf, 3, 1, 1)),
    ("post_upsample", "meta"):              lambda in_ch, nf, scale: (_IdentityPre(), nn.Conv2d(in_ch, nf, 3, 1, 1)),

    ("pre_upsample",  "bicubic"):           lambda in_ch, nf, scale: (_BicubicPre(), nn.Conv2d(in_ch, nf, 3, 1, 1)),
    ("pre_upsample",  "learnable"):         lambda in_ch, nf, scale: (_LearnablePre(in_ch, nf, scale), nn.Identity()),
    ("pre_upsample",  "carafe"):            lambda in_ch, nf, scale: (_CARAFEPre(in_ch, scale), nn.Conv2d(in_ch, nf, 3, 1, 1)),
}

# (mode, method) -> tail module factory
TAIL_BUILDERS: Dict[Tuple[str, str], Callable[[int,int,int], nn.Module]] = {
    ("post_upsample", "pixelshuffle"):      lambda scale, nf, out_ch: _UpsamplerPixelShuffle(scale, nf, out_ch),
    ("post_upsample", "deconvolution"):     lambda scale, nf, out_ch: _UpsamplerDeconv(scale, nf, out_ch),
    ("post_upsample", "pixelshuffle_icnr"): lambda scale, nf, out_ch: _UpsamplerPixelShuffleICNR(scale, nf, out_ch),
    ("post_upsample", "meta"):              lambda scale, nf, out_ch: _MetaUp2d(nf, scale=scale, out_ch=out_ch),

    ("pre_upsample",  "bicubic"):           lambda scale, nf, out_ch: nn.Conv2d(nf, out_ch, 3, 1, 1),
    ("pre_upsample",  "learnable"):         lambda scale, nf, out_ch: nn.Conv2d(nf, out_ch, 3, 1, 1),
    ("pre_upsample",  "carafe"):            lambda scale, nf, out_ch: nn.Conv2d(nf, out_ch, 3, 1, 1),
}

def _validate(mode: str, method: str) -> None:
    if mode not in SUPPORTED:
        raise ValueError(f"Unsupported mode='{mode}'. Supported: {list(SUPPORTED.keys())}")
    if method not in SUPPORTED[mode]:
        raise ValueError(f"Unsupported method='{method}' for mode='{mode}'. Supported: {SUPPORTED[mode]}")
    if (mode, method) not in HEAD_BUILDERS or (mode, method) not in TAIL_BUILDERS:
        raise ValueError(f"No builder registered for (mode={mode}, method={method}).")

# -------------------- model --------------------
class EDSR(nn.Module):
    """
    EDSR-Lite SR (refactor)
    - Input:  [B, C, h, w]
    - Output: [B, C, H, W] với H = h * scale và W = w * scale
    """
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        nf: int = 64,
        nb: int = 12,
        res_scale: float = 0.1,
        scale: int = 2,
        mode: str = "post_upsample",
        method: str = "pixelshuffle",
    ) -> None:
        super().__init__()
        _validate(mode, method)

        self.in_ch, self.out_ch = in_ch, out_ch
        self.nf, self.nb = nf, nb
        self.res_scale = res_scale
        self.scale = int(scale)
        self.mode, self.method = mode, method

        # choose strategies once
        self.pre, self.head = HEAD_BUILDERS[(mode, method)](in_ch, nf, self.scale)
        self.body = nn.Sequential(*[_ResBlock(nf, res_scale) for _ in range(nb)])
        self.tail = TAIL_BUILDERS[(mode, method)](self.scale, nf, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != self.in_ch:
            raise ValueError(f"Input must be [B,{self.in_ch},h,w], got {tuple(x.shape)}")

        b, c, h, w = x.shape
        target_hw = (h * self.scale, w * self.scale)

        # unified path
        s_in = self.pre(x, target_hw)   # identity / bicubic / learnable / carafe
        s = self.head(s_in)             # Conv2d hoặc Identity
        y = self.body(s)
        y = self.tail(y + s)            # upsample tail hoặc 1× conv tại HR

        # size guard
        if (y.size(-2), y.size(-1)) != target_hw:
            y = F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)
        return y.clamp(0, 1)
