from typing import Dict, Tuple, Optional
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


def _make_vgg_feature_extractor(layer: str = "features_35") -> nn.Module:
    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
    # Freeze
    for p in vgg.parameters():
        p.requires_grad = False
    vgg.eval()
    # pick up to relu5_4 approx (features[35]) by default
    upto = int(layer.split("_")[-1]) if layer.startswith("features_") else 35
    return nn.Sequential(*list(vgg.children())[: upto + 1])


class _PatchDiscriminator(nn.Module):
    """A lightweight PatchGAN-style discriminator for SR."""

    def __init__(self, in_ch: int = 3, base_ch: int = 32) -> None:
        super().__init__()
        c = base_ch
        layers = [
            nn.Conv2d(in_ch, c, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, 2 * c, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * c, 2 * c, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2 * c, 4 * c, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * c, 4 * c, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4 * c, 1, 3, 1, 1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ESRGANTrainer:
    """Adversarial trainer for ESRGAN-like SR (compact).

    Losses:
    - L1 pixel loss
    - Perceptual (VGG feature) loss
    - Relativistic average GAN (RaGAN) loss (BCE with logits)
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        model_g: Optional[nn.Module] = None,
        device: str = "cuda",
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
        epochs: int = 50,
        ckpt_g: str = "g_best.pt",
        ckpt_d: str = "d_best.pt",
        w_pix: float = 1.0,
        w_perc: float = 1.0,
        w_gan: float = 0.005,
        disc_base_ch: int = 32,
    ) -> None:
        self.device = device
        # Accept both 'model' (Hydra from main.py) or 'model_g'
        g = model_g if model_g is not None else model
        if g is None:
            raise TypeError("ESRGANTrainer requires a generator passed as 'model' or 'model_g'")
        self.g = g.to(device)
        self.d = _PatchDiscriminator(in_ch=3, base_ch=disc_base_ch).to(device)
        self.opt_g = torch.optim.Adam(self.g.parameters(), lr=lr_g, betas=(beta1, beta2))
        self.opt_d = torch.optim.Adam(self.d.parameters(), lr=lr_d, betas=(beta1, beta2))
        self.sch_g = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_g, T_max=epochs)
        self.sch_d = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_d, T_max=epochs)
        self.epochs = epochs
        self.ckpt_g, self.ckpt_d = ckpt_g, ckpt_d
        self.best = -1e9

        # losses
        self.pix_loss = nn.L1Loss()
        self.vgg = _make_vgg_feature_extractor("features_35").to(device)
        self.w_pix, self.w_perc, self.w_gan = w_pix, w_perc, w_gan
        self.bce = nn.BCEWithLogitsLoss()

    def _vgg_features(self, x: torch.Tensor) -> torch.Tensor:
        # normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.vgg(x)

    def _gan_d_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        # Relativistic average GAN loss
        real_rel = real_logits - fake_logits.mean()
        fake_rel = fake_logits - real_logits.mean()
        loss_real = self.bce(real_rel, torch.ones_like(real_rel))
        loss_fake = self.bce(fake_rel, torch.zeros_like(fake_rel))
        return (loss_real + loss_fake) * 0.5

    def _gan_g_loss(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        real_rel = real_logits - fake_logits.mean()
        fake_rel = fake_logits - real_logits.mean()
        return self.bce(fake_rel, torch.ones_like(fake_rel))

    def fit(self, train_loader, val_fn) -> None:
        for ep in range(1, self.epochs + 1):
            self.g.train(); self.d.train()
            t0 = time.time(); loss_g_sum = 0.0; loss_d_sum = 0.0; n_items = 0

            for lr_img, hr_img in train_loader:
                lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
                b = lr_img.size(0)
                n_items += b

                # 1) Train D
                with torch.no_grad():
                    sr = self.g(lr_img)
                real_logits = self.d(hr_img)
                fake_logits = self.d(sr.detach())
                loss_d = self._gan_d_loss(real_logits, fake_logits)
                self.opt_d.zero_grad(set_to_none=True); loss_d.backward(); self.opt_d.step()

                # 2) Train G
                sr = self.g(lr_img)
                fake_logits = self.d(sr)
                real_logits = self.d(hr_img.detach())
                # pixel loss
                l_pix = self.pix_loss(sr, hr_img)
                # perceptual loss
                f_sr = self._vgg_features(sr)
                f_hr = self._vgg_features(hr_img)
                l_perc = F.l1_loss(f_sr, f_hr)
                # gan loss
                l_gan = self._gan_g_loss(real_logits, fake_logits)

                loss_g = self.w_pix * l_pix + self.w_perc * l_perc + self.w_gan * l_gan
                self.opt_g.zero_grad(set_to_none=True); loss_g.backward(); self.opt_g.step()

                loss_g_sum += loss_g.item() * b
                loss_d_sum += loss_d.item() * b

            self.sch_g.step(); self.sch_d.step()
            val_metrics: Dict[str, float] = val_fn(self.g)
            psnr = val_metrics["psnr"]
            print(
                f"[ESRGAN {ep:03d}] G={loss_g_sum/n_items:.4f} D={loss_d_sum/n_items:.4f} | "
                f"Val PSNR={psnr:.3f} SSIM={val_metrics['ssim']:.4f} | "
                f"lrG={self.sch_g.get_last_lr()[0]:.2e} lrD={self.sch_d.get_last_lr()[0]:.2e}"
            )
            if psnr > self.best:
                self.best = psnr
                torch.save(self.g.state_dict(), self.ckpt_g)
                torch.save(self.d.state_dict(), self.ckpt_d)

    def load_best(self) -> None:
        self.g.load_state_dict(torch.load(self.ckpt_g, map_location=self.device))
