# src/eval/evaluator.py
from typing import Dict, Optional
import torch
import torch.nn as nn
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
                plt.figure(figsize=(12, 3))
                for j, (img, title) in enumerate([(lr[i], "LR"), (sr[i], "SR"), (hr[i], "HR")]):
                    plt.subplot(1, 3, j + 1)
                    plt.imshow(img.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy())
                    plt.title(title); plt.axis("off")
                plt.tight_layout(); plt.show()
                shown += 1
