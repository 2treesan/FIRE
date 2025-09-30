import torch
import torch.nn as nn
import torch.nn.functional as F

class _Act(nn.Module):
    def __init__(self, kind: str = "prelu"):
        super().__init__()
        if kind not in ["relu", "prelu"]:
            raise ValueError("activation must be 'relu' or 'prelu'")
        self.kind = kind
        self.act = nn.PReLU() if kind == "prelu" else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x)

class FSRCNN(nn.Module):
    """
    FSRCNN (post-upsampling)
    Input:  [B, C, h, w]
    Output: [B, C, H, W] with H = h * scale, W = w * scale
    """
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        num_feats: int = 56,
        d: int = 12,
        s: int = 4,
        m: int = 4,
        scale: int = 2,
        activation: str = "prelu",
    ) -> None:
        super().__init__()
        if scale not in (2, 3, 4):
            # keep same supported scales as classical FSRCNN
            raise ValueError("FSRCNN supports scale in {2,3,4}")
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.scale = int(scale)

        Act = lambda: _Act(activation)

        # feature extraction (5x5)
        self.feature = nn.Sequential(
            nn.Conv2d(in_ch, num_feats, kernel_size=5, stride=1, padding=2),
            Act(),
        )
        # shrink (1x1)
        self.shrink = nn.Sequential(
            nn.Conv2d(num_feats, d, kernel_size=1, stride=1, padding=0),
            Act(),
        )
        # mapping: m blocks, using kernel 3x3; we adopt d->s mapping then keep s
        mapping_blocks = []
        for i in range(m):
            mapping_blocks.append(nn.Conv2d(d if i==0 else s, s, kernel_size=3, stride=1, padding=1))
            mapping_blocks.append(Act())
        self.mapping = nn.Sequential(*mapping_blocks)

        # expand (1x1) back to num_feats
        self.expand = nn.Sequential(
            nn.Conv2d(s, num_feats, kernel_size=1, stride=1, padding=0),
            Act(),
        )

        # deconv (transpose conv) to upsample to HR
        # kernel_size/padding/output_padding chosen to match output size
        self.deconv = nn.ConvTranspose2d(
            num_feats, out_ch, kernel_size=9, stride=self.scale, padding=4, output_padding=self.scale - 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input validation
        if x.dim() != 4 or x.size(1) != self.in_ch:
            raise ValueError(f"Input must be [B,{self.in_ch},h,w], got {tuple(x.shape)}")
        b, c, h, w = x.shape
        target_h, target_w = h * self.scale, w * self.scale

        y = self.feature(x)
        y = self.shrink(y)
        y = self.mapping(y)
        y = self.expand(y)
        y = self.deconv(y)

        # ensure final exact size (guard small off-by-one)
        if y.size(-2) != target_h or y.size(-1) != target_w:
            y = F.interpolate(y, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return y.clamp(0, 1)
