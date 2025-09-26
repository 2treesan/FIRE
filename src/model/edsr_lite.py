import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_MODE = {
    # mode : [<list of supported methods>]
    "pre_upsample": ["learnable", "bicubic"],
    "post_upsample": ["deconvolution", "pixelshuffle"],
}

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
            # out = (in-1)*2 - 2*1 + 4 = 2*in
            m += [nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1)]
        elif scale == 3:
            # out = (in-1)*3 - 2*2 + 7 = 3*in
            m += [nn.ConvTranspose2d(nf, nf, kernel_size=7, stride=3, padding=2)]
        elif scale == 4:
            # two stages of x2
            m += [nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1),
                  nn.ReLU(inplace=True),
                  nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1)]
        else:
            raise ValueError(f"Unsupported scale={scale}")
        m += [nn.Conv2d(nf, out_ch, 3, 1, 1)]
        super().__init__(*m)

class EDSRLite(nn.Module):
    """
    EDSR-Lite SR
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

        if mode not in SUPPORTED_MODE:
            raise ValueError(f"Unsupported mode='{mode}'. Supported: {list(SUPPORTED_MODE.keys())}")
        if method not in SUPPORTED_MODE[mode]:
            raise ValueError(
                f"Unsupported method='{method}' for mode='{mode}'. "
                f"Supported: {SUPPORTED_MODE[mode]}"
            )

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nf = nf
        self.nb = nb
        self.res_scale = res_scale
        self.scale = int(scale)
        self.mode = mode
        self.method = method

        # Core body (residual blocks) always operate on 'nf' channels
        self.body = nn.Sequential(*[_ResBlock(nf, res_scale) for _ in range(nb)])

        if self.mode == "post_upsample":
            # Work in LR space, upsample at tail using selected method
            self.head = nn.Conv2d(in_ch, nf, 3, 1, 1)
            if self.method == "pixelshuffle":
                self.tail = _UpsamplerPixelShuffle(self.scale, nf, out_ch)
            elif self.method == "deconvolution":
                self.tail = _UpsamplerDeconv(self.scale, nf, out_ch)
            else:  # pragma: no cover (guarded by validation)
                raise RuntimeError("Invalid method for post_upsample")

        elif self.mode == "pre_upsample":
            # Upsample to HR first, then do 1x restoration in HR space
            if self.method == "bicubic":
                # Bicubic pre-upsample in image space
                self.pre = None  # handled in forward via F.interpolate
                self.head = nn.Conv2d(in_ch, nf, 3, 1, 1)
            elif self.method == "learnable":
                # Learnable pre-upsample features to HR using PixelShuffle
                # Conv maps RGB->nf*(s^2), then PixelShuffle -> [B,nf,H,W]
                self.pre = nn.Sequential(
                    nn.Conv2d(in_ch, nf * (self.scale ** 2), 3, 1, 1),
                    nn.PixelShuffle(self.scale),
                )
                self.head = nn.Identity()  # pre already yields nf at HR
            else:  # pragma: no cover (guarded by validation)
                raise RuntimeError("Invalid method for pre_upsample")
            self.tail = nn.Conv2d(nf, out_ch, 3, 1, 1)  # 1× at HR

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Validate input shape
        if x.dim() != 4 or x.size(1) != self.in_ch:
            raise ValueError(
                f"Input must be [B,{self.in_ch},h,w], got {tuple(x.shape)}"
            )
        b, c, h, w = x.shape
        target_h, target_w = h * self.scale, w * self.scale

        if self.mode == "post_upsample":
            s = self.head(x)  # [B,nf,h,w]
            y = self.body(s)
            y = self.tail(y + s)  # upsample to [B,out_ch,H,W]
        else:  # pre_upsample
            if self.method == "bicubic":
                x_hr = F.interpolate(
                    x, size=(target_h, target_w), mode="bicubic", align_corners=False
                )
                s = self.head(x_hr)  # [B,nf,H,W]
            else:  # learnable
                s = self.pre(x)  # [B,nf,H,W]
            y = self.body(s)
            y = self.tail(y + s)  # 1x conv at HR

        # Ensure output spatial size exactly matches (H,W)
        if y.size(-2) != target_h or y.size(-1) != target_w:
            y = F.interpolate(y, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return y.clamp(0, 1)