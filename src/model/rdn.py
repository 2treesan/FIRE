import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class _RDB_Conv(nn.Module):
	"""
	One convolution layer inside an RDB with dense connection.
	in:  [B, in_ch, H, W]
	out: [B, in_ch + growth_rate, H, W] (concat with input)
	"""

	def __init__(self, in_ch: int, growth_rate: int) -> None:
		super().__init__()
		self.conv = nn.Conv2d(in_ch, growth_rate, kernel_size=3, stride=1, padding=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = F.relu(self.conv(x), inplace=True)
		return torch.cat((x, out), dim=1)


class _RDB(nn.Module):
	"""
	Residual Dense Block (RDB)
	- C layers with dense connections (growth_rate channels each)
	- Local Feature Fusion (1x1 conv) to compress back to G0
	- Local Residual Learning: add block input
	"""

	def __init__(self, G0: int, C: int, G: int) -> None:
		super().__init__()
		layers: List[nn.Module] = []
		in_ch = G0
		for _ in range(C):
			layers.append(_RDB_Conv(in_ch, G))
			in_ch += G
		self.dense_layers = nn.Sequential(*layers)
		self.lff = nn.Conv2d(G0 + C * G, G0, kernel_size=1, stride=1, padding=0)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		dense_out = self.dense_layers(x)
		lff_out = self.lff(dense_out)
		return lff_out + x


class _UpsamplerPixelShuffle(nn.Sequential):
	"""PixelShuffle upsampler for scale 2/3/4 (EDSR/RDN style)."""

	def __init__(self, scale: int, nf: int, out_ch: int) -> None:
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


class RDN(nn.Module):
	"""
	Residual Dense Network for Image Super-Resolution (CVPR 2018)

	Single canonical configuration following the paper/default practice:
	- G0 (feature channels) = 64
	- D  (number of RDBs)   = 20
	- C  (conv per RDB)     = 6
	- G  (growth rate)      = 32

	Input : [B, in_ch, h, w]
	Output: [B, out_ch, H, W] with H = h * scale, W = w * scale
	"""

	def __init__(
		self,
		in_ch: int = 3,
		out_ch: int = 3,
		G0: int = 64,  # feature channels
		D: int = 20,   # number of RDBs
		C: int = 6,    # conv layers per RDB
		G: int = 32,   # growth rate inside RDB
		scale: int = 2,
	) -> None:
		super().__init__()
		if scale not in (2, 3, 4):
			raise ValueError(f"RDN only supports scale 2/3/4, got {scale}")

		self.in_ch = in_ch
		self.out_ch = out_ch
		self.G0 = G0
		self.D = D
		self.C = C
		self.G = G
		self.scale = int(scale)

		# Shallow feature extraction
		self.sfe1 = nn.Conv2d(in_ch, G0, kernel_size=3, stride=1, padding=1)
		self.sfe2 = nn.Conv2d(G0, G0, kernel_size=3, stride=1, padding=1)

		# Residual Dense Blocks
		self.rdbs = nn.ModuleList([_RDB(G0=G0, C=C, G=G) for _ in range(D)])

		# Global Feature Fusion (concat RDB outputs -> 1x1 -> 3x3)
		self.gff_1x1 = nn.Conv2d(D * G0, G0, kernel_size=1, stride=1, padding=0)
		self.gff_3x3 = nn.Conv2d(G0, G0, kernel_size=3, stride=1, padding=1)

		# Upsampler (PixelShuffle)
		self.upsampler = _UpsamplerPixelShuffle(self.scale, nf=G0, out_ch=out_ch)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.dim() != 4 or x.size(1) != self.in_ch:
			raise ValueError(f"Input must be [B,{self.in_ch},h,w], got {tuple(x.shape)}")

		b, c, h, w = x.shape
		target_hw = (h * self.scale, w * self.scale)

		# Shallow features
		f_m1 = self.sfe1(x)
		f0 = self.sfe2(f_m1)

		# RDB trunk
		rdb_outs: List[torch.Tensor] = []
		f = f0
		for rdb in self.rdbs:
			f = rdb(f)
			rdb_outs.append(f)

		# Global feature fusion + global residual
		trunk_concat = torch.cat(rdb_outs, dim=1)
		gff = self.gff_3x3(self.gff_1x1(trunk_concat))
		F_DF = gff + f_m1

		y = self.upsampler(F_DF)

		# size guard (just in case)
		if (y.size(-2), y.size(-1)) != target_hw:
			y = F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)

		return y.clamp(0, 1)

