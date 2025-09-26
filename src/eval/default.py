# src/eval/evaluator.py
from typing import Dict, Optional
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
from .postprocess import build_postprocess

class DefaultEvaluator:
    def __init__(self, loader, device: str = "cuda",
                 postprocess: str = "identity",   # model đã LR->HR sẵn
                 n_samples_show: int = 4) -> None:
        self.loader = loader
        self.device = device
        self.post_name = postprocess
        self.post = None 
        self.n_samples_show = n_samples_show
        # torchmetrics
        self.psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def set_model(self, model: nn.Module) -> None:
        """Gọi sau khi đã có model; sẽ khởi tạo hậu xử lý theo tên."""
        self.post = build_postprocess(self.post_name, model)

    @torch.no_grad()
    def __call__(self, model: nn.Module) -> Dict[str, float]:
        if self.post is None:
            self.set_model(model)

        model.eval()
        ps_sum, ss_sum, n = 0.0, 0.0, 0
        for lr, hr in self.loader:
            lr, hr = lr.to(self.device), hr.to(self.device)
            sr = self.post(lr)  # đã bọc model trong post
            ps = self.psnr_metric(sr, hr).item()
            ss = self.ssim_metric(sr, hr).item()
            ps_sum += ps; ss_sum += ss; n += 1
        return {"psnr": ps_sum / n, "ssim": ss_sum / n}

    @torch.no_grad()
    def show_samples(self, model: nn.Module, n: Optional[int] = None) -> None:
        if self.post is None:
            self.set_model(model)
        if n is None:
            n = self.n_samples_show

        model.eval()
        shown = 0
        for lr, hr in self.loader:
            lr, hr = lr.to(self.device), hr.to(self.device)
            sr = self.post(lr)
            b = lr.size(0)
            for i in range(b):
                if shown >= n:
                    return
                # Prepare a grid: first row = full images; next rows = cropped comparisons
                num_crops = 5
                rows, cols = 1 + num_crops, 3
                fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * (1 + num_crops)))
                if rows == 1:
                    # edge case, though we always have num_crops=5
                    axes = [axes]

                def show(ax, img_tensor, title: str):
                    ax.imshow(img_tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
                    ax.set_title(title)
                    ax.axis("off")

                # Row 0: full images
                show(axes[0][0], lr[i], "LR")
                show(axes[0][1], sr[i], "SR")
                show(axes[0][2], hr[i], "HR")

                # Compute scale between LR and HR to align crops
                _, lr_h, lr_w = lr[i].shape
                _, sr_h, sr_w = sr[i].shape
                _, hr_h, hr_w = hr[i].shape
                # Prefer SR dims for target (should match HR for SR tasks)
                t_h, t_w = sr_h, sr_w
                scale_h = t_h / max(1, lr_h)
                scale_w = t_w / max(1, lr_w)

                # Choose a reasonable LR crop size (square)
                min_lr_dim = min(lr_h, lr_w)
                crop_lr = max(16, min_lr_dim // 4)
                crop_lr = min(crop_lr, lr_h, lr_w)  # ensure fits

                # Build an information score map on LR (detail + colorfulness)
                def _normalize(t: torch.Tensor) -> torch.Tensor:
                    t = t - t.min()
                    den = t.max().clamp_min(1e-8)
                    return t / den

                def _score_map(img_cxhxw: torch.Tensor) -> torch.Tensor:
                    # luminance for edges
                    if img_cxhxw.size(0) >= 3:
                        r, g, b = img_cxhxw[0], img_cxhxw[1], img_cxhxw[2]
                        y = 0.299 * r + 0.587 * g + 0.114 * b
                        color_std = img_cxhxw[:3].std(dim=0)
                    else:
                        y = img_cxhxw[0]
                        color_std = torch.zeros_like(y)
                    y4 = y.unsqueeze(0).unsqueeze(0)
                    kx = torch.tensor([[-1, 0, 1],
                                       [-2, 0, 2],
                                       [-1, 0, 1]], dtype=img_cxhxw.dtype, device=img_cxhxw.device).view(1, 1, 3, 3)
                    ky = torch.tensor([[-1, -2, -1],
                                       [ 0,  0,  0],
                                       [ 1,  2,  1]], dtype=img_cxhxw.dtype, device=img_cxhxw.device).view(1, 1, 3, 3)
                    gx = F.conv2d(y4, kx, padding=1)
                    gy = F.conv2d(y4, ky, padding=1)
                    grad = gx.abs() + gy.abs()
                    grad = grad[0, 0]
                    s = _normalize(grad) + 0.5 * _normalize(color_std)
                    return s

                def _select_topk(score_hw: torch.Tensor, k: int, crop: int):
                    # average pooling to score crops; stride=1 gives top-left indices
                    pooled = F.avg_pool2d(score_hw.unsqueeze(0).unsqueeze(0), crop, stride=1)
                    pooled = pooled[0, 0].clone()
                    coords = []
                    if pooled.numel() == 0:
                        return coords
                    suppress = max(1, int(round(crop * 0.5)))
                    for _ in range(k):
                        flat_idx = torch.argmax(pooled.view(-1)).item()
                        y = flat_idx // pooled.size(1)
                        x = flat_idx % pooled.size(1)
                        coords.append((int(y), int(x)))
                        # suppress neighborhood to avoid heavy overlap
                        y0 = max(0, y - suppress)
                        y1 = min(pooled.size(0), y + suppress + 1)
                        x0 = max(0, x - suppress)
                        x1 = min(pooled.size(1), x + suppress + 1)
                        pooled[y0:y1, x0:x1] = -1e9
                    return coords

                score = _score_map(lr[i])
                top_lefts = _select_topk(score, num_crops, crop_lr)

                # Generate and show informative crops
                for k, (y_lr, x_lr) in enumerate(top_lefts):
                    # LR crop (as-is)
                    lr_crop = lr[i, :, y_lr:y_lr + crop_lr, x_lr:x_lr + crop_lr]

                    # Map to SR/HR coordinates
                    y_t = int(round(y_lr * scale_h))
                    x_t = int(round(x_lr * scale_w))
                    h_t = int(round(crop_lr * scale_h))
                    w_t = int(round(crop_lr * scale_w))
                    # Clamp to target bounds, adjust starts if needed to keep size
                    y_t = max(0, min(t_h - h_t, y_t)) if h_t <= t_h else 0
                    x_t = max(0, min(t_w - w_t, x_t)) if w_t <= t_w else 0
                    y_t2 = min(t_h, y_t + h_t)
                    x_t2 = min(t_w, x_t + w_t)

                    sr_crop = sr[i, :, y_t:y_t2, x_t:x_t2]
                    # For HR, try to crop with same target dims; fall back to HR dims if SR differs
                    hr_y_t = max(0, min(hr_h - (y_t2 - y_t), y_t))
                    hr_x_t = max(0, min(hr_w - (x_t2 - x_t), x_t))
                    hr_crop = hr[i, :, hr_y_t:hr_y_t + (y_t2 - y_t), hr_x_t:hr_x_t + (x_t2 - x_t)]

                    row = k + 1
                    show(axes[row][0], lr_crop, f"LR crop {k+1}")
                    show(axes[row][1], sr_crop, f"SR crop {k+1}")
                    show(axes[row][2], hr_crop, f"HR crop {k+1}")

                plt.tight_layout()
                plt.show()
                shown += 1
