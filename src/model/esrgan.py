import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class _ResidualDenseBlock5C(nn.Module):
	"""Residual Dense Block with 5 convs (RDB-5C) used in RRDBNet/ESRGAN.
	Uses dense connections inside the block and local residual scaling.
	"""

	def __init__(self, nf: int, gc: int = 32, res_scale: float = 0.2) -> None:
		super().__init__()
		self.res_scale = res_scale
		self.c1 = nn.Conv2d(nf, gc, 3, 1, 1)
		self.c2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
		self.c3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
		self.c4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
		self.c5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x1 = F.leaky_relu(self.c1(x), negative_slope=0.2, inplace=True)
		x2 = F.leaky_relu(self.c2(torch.cat([x, x1], dim=1)), 0.2, inplace=True)
		x3 = F.leaky_relu(self.c3(torch.cat([x, x1, x2], dim=1)), 0.2, inplace=True)
		x4 = F.leaky_relu(self.c4(torch.cat([x, x1, x2, x3], dim=1)), 0.2, inplace=True)
		x5 = self.c5(torch.cat([x, x1, x2, x3, x4], dim=1))
		return x + self.res_scale * x5


class _RRDB(nn.Module):
	"""Residual in Residual Dense Block (RRDB)."""

	def __init__(self, nf: int, gc: int, res_scale: float = 0.2) -> None:
		super().__init__()
		self.res_scale = res_scale
		self.b1 = _ResidualDenseBlock5C(nf, gc, res_scale)
		self.b2 = _ResidualDenseBlock5C(nf, gc, res_scale)
		self.b3 = _ResidualDenseBlock5C(nf, gc, res_scale)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.b1(x)
		out = self.b2(out)
		out = self.b3(out)
		return x + self.res_scale * out


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


class ESRGAN(nn.Module):
	"""Compact ESRGAN-like Generator (RRDBNet backbone) for SR.
	Kept small via nf/gc/nb reductions but preserves RRDB structure.
	"""

	def __init__(
		self,
		in_ch: int = 3,
		out_ch: int = 3,
		nf: int = 48,          # base feature channels (smaller than typical 64/64+)
		nb: int = 8,           # number of RRDB blocks (smaller than typical 23)
		gc: int = 16,          # growth channels inside RDB (smaller than 32)
		res_scale: float = 0.2,
		scale: int = 2,
	) -> None:
		super().__init__()
		if scale not in (2, 3, 4):
			raise ValueError(f"ESRGAN_G only supports scale 2/3/4, got {scale}")
		self.in_ch, self.out_ch = in_ch, out_ch
		self.nf, self.nb, self.gc = nf, nb, gc
		self.res_scale, self.scale = res_scale, int(scale)

		# head
		self.head = nn.Conv2d(in_ch, nf, 3, 1, 1)

		# trunk (RRDB blocks)
		blocks = [_RRDB(nf, gc, res_scale) for _ in range(nb)]
		self.trunk = nn.Sequential(*blocks)
		self.trunk_tail = nn.Conv2d(nf, nf, 3, 1, 1)

		# upsampler and final conv
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

