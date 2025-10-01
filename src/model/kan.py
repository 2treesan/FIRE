import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class _KANAct(nn.Module):
	"""
	Lightweight Kolmogorov–Arnold inspired activation for CNN features.
	Per-channel elementwise activation as a learnable mixture of Gaussian basis
	functions plus a linear term: a_c * x + b_c + sum_k w_{c,k} * exp(-(x - t_k)^2 / (2*sigma^2)).

	- Shared knots t_k across channels.
	- Per-channel weights a_c, b_c and w_{c,k}.
	"""

	def __init__(self, channels: int, n_bases: int = 8, sigma: float = 0.5) -> None:
		super().__init__()
		self.channels = channels
		self.n_bases = n_bases
		self.register_buffer("knots", torch.linspace(-2.0, 2.0, steps=n_bases))
		self.sigma = float(sigma)
		# per-channel linear params
		self.a = nn.Parameter(torch.ones(channels))
		self.b = nn.Parameter(torch.zeros(channels))
		# per-channel basis weights [C, K]
		self.w = nn.Parameter(torch.zeros(channels, n_bases))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, C, H, W]
		b, c, h, w = x.shape
		x_lin = self.a.view(1, c, 1, 1) * x + self.b.view(1, c, 1, 1)
		# RBF basis evaluation
		# Compute for each k: phi_k(x) = exp(-(x - t_k)^2 / (2*sigma^2))
		sigma2 = 2.0 * (self.sigma ** 2)
		out = x_lin
		for k in range(self.n_bases):
			tk = self.knots[k]
			phi = torch.exp(-((x - tk) ** 2) / sigma2)
			wk = self.w[:, k].view(1, c, 1, 1)
			out = out + wk * phi
		return out


class _KANResBlock(nn.Module):
	def __init__(self, nf: int, n_bases: int = 8, sigma: float = 0.5, res_scale: float = 0.1) -> None:
		super().__init__()
		self.c1 = nn.Conv2d(nf, nf, 3, 1, 1)
		self.act = _KANAct(nf, n_bases=n_bases, sigma=sigma)
		self.c2 = nn.Conv2d(nf, nf, 3, 1, 1)
		self.res_scale = res_scale

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		y = self.c2(self.act(self.c1(x)))
		return x + self.res_scale * y


def _icnr_(w: torch.Tensor, scale: int, init=nn.init.kaiming_normal_):
	"""ICNR init for PixelShuffle weights to reduce checkerboard artifacts."""
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
		self.scale = int(scale)
		if self.scale in (2, 3):
			self.conv = nn.Conv2d(nf, nf * (self.scale ** 2), 3, 1, 1)
			self.ps = nn.PixelShuffle(self.scale)
			self.tail = nn.Conv2d(nf, out_ch, 3, 1, 1)
			_icnr_(self.conv.weight, scale=self.scale)
		elif self.scale == 4:
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
			return self.tail(self.ps(self.conv(x)))
		else:
			x = self.ps1(self.conv1(x))
			x = self.ps2(self.conv2(x))
			return self.tail(x)


class KANEDSR(nn.Module):
	"""
	Hybrid KAN + EDSR for Image Super-Resolution.

	- EDSR-style shallow head and tail with PixelShuffle+ICNR (post upsample)
	- Residual body uses KANResBlocks: Conv -> KANAct (RBF spline) -> Conv with residual
	- Global residual like EDSR
	"""

	def __init__(
		self,
		in_ch: int = 3,
		out_ch: int = 3,
		nf: int = 64,
		nb: int = 12,
		res_scale: float = 0.1,
		scale: int = 2,
		kan_bases: int = 8,
		kan_sigma: float = 0.5,
	) -> None:
		super().__init__()
		if scale not in (2, 3, 4):
			raise ValueError(f"KANEDSR only supports scale 2/3/4, got {scale}")

		self.in_ch, self.out_ch = in_ch, out_ch
		self.nf, self.nb = nf, nb
		self.res_scale = res_scale
		self.scale = int(scale)

		# head
		self.head = nn.Conv2d(in_ch, nf, 3, 1, 1)

		# body (KAN residual blocks)
		body: List[nn.Module] = [
			_KANResBlock(nf, n_bases=kan_bases, sigma=kan_sigma, res_scale=res_scale)
			for _ in range(nb)
		]
		self.body = nn.Sequential(*body)
		self.body_tail = nn.Conv2d(nf, nf, 3, 1, 1)

		# tail (upsampler)
		self.upsampler = _UpsamplerPixelShuffleICNR(self.scale, nf=nf, out_ch=out_ch)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if x.dim() != 4 or x.size(1) != self.in_ch:
			raise ValueError(f"Input must be [B,{self.in_ch},h,w], got {tuple(x.shape)}")

		b, c, h, w = x.shape
		target_hw = (h * self.scale, w * self.scale)

		s = self.head(x)
		y = self.body_tail(self.body(s))
		y = y + s
		y = self.upsampler(y)

		if (y.size(-2), y.size(-1)) != target_hw:
			y = F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)
		return y.clamp(0, 1)


__all__ = ["KANEDSR"]

