# src/loader/default.py
from typing import Tuple, List, Sequence, Optional
from glob import glob
import os, random
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .augment import build_pipeline as build_aug

def _bicubic():
    return getattr(getattr(Image, "Resampling", Image), "BICUBIC")

def _mod_crop(img: Image.Image, scale: int) -> Image.Image:
    w, h = img.size
    w2, h2 = (w // scale) * scale, (h // scale) * scale
    if w2 == w and h2 == h: return img
    return img.crop((0, 0, w2, h2))  # top-left anchored (chuẩn SR)

def _hr2lr(hr: Image.Image, scale: int) -> Image.Image:
    w, h = hr.size
    hr = hr.filter(ImageFilter.GaussianBlur(radius= 8 / scale))
    return hr.resize((w // scale, h // scale), resample=_bicubic())

class _HROnlyDataset(Dataset):
    def __init__(self,
                 hr_paths: Sequence[str],
                 scale: int,
                 phase: str,
                 aug = None,
                 patch_size: int = 128):            # train patch size trên HR (vuông)
        assert len(hr_paths) > 0
        self.hr_paths = list(hr_paths)
        self.scale = int(scale)
        self.phase = phase
        self.aug = aug
        # enforce patch_size divisible by scale strictly (no silent rounding)
        patch_size = int(patch_size)
        assert (patch_size == 0) or (patch_size % self.scale == 0), \
            f"patch_size ({patch_size}) must be divisible by scale ({self.scale}) or 0"
        self.patch_size = patch_size
        self.to_tensor = transforms.ToTensor()

    def _random_hr_patch(self, hr: Image.Image) -> Image.Image:
        ps = self.patch_size
        if ps <= 0:  # không crop
            return hr
        w, h = hr.size
        if ps > min(w, h):
            # fallback: lấy center crop nhỏ nhất có thể chia hết scale
            ps2 = min(w, h) // self.scale * self.scale
            ps = max(self.scale, ps2)
        x = random.randint(0, w - ps); y = random.randint(0, h - ps)
        return hr.crop((x, y, x + ps, y + ps))

    # center-crop removed per request to avoid val-time augmentation

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, i: int):
        hr_full = Image.open(self.hr_paths[i]).convert("RGB")

        if self.phase == "train":
            hr_full = _mod_crop(hr_full, self.scale)
            hr = self._random_hr_patch(hr_full) if self.patch_size > 0 else hr_full
        else:
            hr = _mod_crop(hr_full, self.scale)

        lr = _hr2lr(hr, self.scale)  # LR nhỏ hơn HR theo scale

        # Augment sau khi đã cặp (lr, hr) chỉ áp dụng cho phase==train để tránh thay đổi val/test
        if self.aug is not None and self.phase == "train":
            lr, hr = self.aug(lr, hr)

        return self.to_tensor(lr), self.to_tensor(hr)

class DefaultLoader:
    def __init__(self, hr_dir: str, split = (0.8, 0.1, 0.1),
                 batch_size = 16, num_workers = 4, seed = 42,
                 exts = (".png", ".jpg", ".jpeg"),
                 augment: Optional[List[str]] = None,
                 scale: int = 2,
                 patch_size: int = 128):
        self.hr_dir = hr_dir
        self.split = split
        self.batch_size, self.num_workers = batch_size, num_workers
        self.seed = seed; self.exts = exts
        # single list of augment names, applied only on train split
        self.augment = augment or []
        self.scale = int(scale)
        self.patch_size = int(patch_size)
        # enforce at loader level too for early failure
        assert (self.patch_size == 0) or (self.patch_size % self.scale == 0), \
            f"patch_size ({self.patch_size}) must be divisible by scale ({self.scale}) or 0"

    def _list_images(self, d: str) -> List[str]:
        xs: List[str] = []
        for e in self.exts: xs += glob(os.path.join(d, f"*{e}"))
        return sorted(xs)

    def make(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        hr = self._list_images(self.hr_dir)
        assert len(hr) > 0, f"No images found in {self.hr_dir} with extensions {self.exts}"
        n = len(hr); random.seed(self.seed); ids = list(range(n)); random.shuffle(ids)
        s_tr, s_va, s_te = self.split
        n_te, n_va = int(n * s_te), int(n * s_va); n_tr = n - n_va - n_te
        pick = lambda a, I: [a[i] for i in I]
        id_tr, id_va, id_te = ids[:n_tr], ids[n_tr:n_tr+n_va], ids[n_tr+n_va:]

        aug_train = build_aug(self.augment) if self.augment else None

        ds_tr = _HROnlyDataset(pick(hr, id_tr), self.scale, "train", aug=aug_train,
                               patch_size=self.patch_size)
        ds_va = _HROnlyDataset(pick(hr, id_va), self.scale, "val",   aug=None,
                               patch_size=0)
        ds_te = _HROnlyDataset(pick(hr, id_te), self.scale, "test",  aug=None,
                               patch_size=0)

        mk = lambda ds, bs, sh: DataLoader(ds, batch_size=bs, shuffle=sh,
                                           num_workers=self.num_workers, pin_memory=True)
        # Important: use batch_size=1 for val/test to handle variable-sized full-resolution images
        return mk(ds_tr, self.batch_size, True), mk(ds_va, 1, False), mk(ds_te, 1, False)
