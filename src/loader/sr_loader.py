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
    """Crop image to be divisible by scale"""
    w, h = img.size
    w2, h2 = (w // scale) * scale, (h // scale) * scale
    if w2 == w and h2 == h: return img
    return img.crop((0, 0, w2, h2))  # top-left anchored

def _hr2lr(hr: Image.Image, scale: int) -> Image.Image:
    """Convert HR to LR by Gaussian blur + bicubic downsampling"""
    w, h = hr.size
    hr = hr.filter(ImageFilter.GaussianBlur(radius=8 / scale))
    return hr.resize((w // scale, h // scale), resample=_bicubic())

class _SRDataset(Dataset):
    def __init__(self,
                 hr_paths: Sequence[str],
                 scale: int,
                 phase: str,
                 aug=None,
                 patch_size: int = 192):  # HR patch size
        assert len(hr_paths) > 0
        self.hr_paths = list(hr_paths)
        self.scale = int(scale)
        self.phase = phase
        self.aug = aug
        # Patch size phải chia hết cho scale
        patch_size = int(patch_size)
        assert (patch_size == 0) or (patch_size % self.scale == 0), \
            f"patch_size ({patch_size}) must be divisible by scale ({self.scale}) or 0"
        self.patch_size = patch_size
        self.to_tensor = transforms.ToTensor()
        
        print(f"Created {phase} dataset with {len(hr_paths)} images")

    def _random_hr_patch(self, hr: Image.Image) -> Image.Image:
        """Random crop HR patch for training"""
        ps = self.patch_size
        if ps <= 0:  # No cropping
            return hr
        w, h = hr.size
        if ps > min(w, h):
            # Fallback: center crop với size nhỏ nhất có thể chia hết cho scale
            ps2 = min(w, h) // self.scale * self.scale
            ps = max(self.scale, ps2)
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)
        return hr.crop((x, y, x + ps, y + ps))

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, i: int):
        hr_full = Image.open(self.hr_paths[i]).convert("RGB")

        if self.phase == "train":
            hr_full = _mod_crop(hr_full, self.scale)
            hr = self._random_hr_patch(hr_full) if self.patch_size > 0 else hr_full
        else:
            # Val/Test: sử dụng toàn bộ ảnh sau khi mod crop
            hr = _mod_crop(hr_full, self.scale)

        lr = _hr2lr(hr, self.scale)

        # Augmentation chỉ cho training
        if self.aug is not None and self.phase == "train":
            lr, hr = self.aug(lr, hr)

        return self.to_tensor(lr), self.to_tensor(hr)

class SRLoader:
    def __init__(self, 
                 hr_dir: str, 
                 split=(0.8, 0.1, 0.1),  # train, val, test
                 batch_size=8, 
                 num_workers=4, 
                 seed=42,
                 exts=(".png", ".jpg", ".jpeg"),
                 augment: Optional[List[str]] = None,
                 scale: int = 4,
                 patch_size: int = 192):  # HR patch size
        self.hr_dir = hr_dir
        self.split = split
        self.batch_size, self.num_workers = batch_size, num_workers
        self.seed = seed
        self.exts = exts
        self.augment = augment or []
        self.scale = int(scale)
        self.patch_size = int(patch_size)
        
        # Validation
        assert (self.patch_size == 0) or (self.patch_size % self.scale == 0), \
            f"patch_size ({self.patch_size}) must be divisible by scale ({self.scale}) or 0"
        assert abs(sum(self.split) - 1.0) < 1e-6, f"Split ratios must sum to 1.0, got {sum(self.split)}"

    def _list_images(self, d: str) -> List[str]:
        """List all images in directory"""
        xs: List[str] = []
        for e in self.exts: 
            xs += glob(os.path.join(d, f"*{e}"))
        return sorted(xs)

    def make(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train, val, test dataloaders"""
        hr_paths = self._list_images(self.hr_dir)
        assert len(hr_paths) > 0, f"No images found in {self.hr_dir} with extensions {self.exts}"
        
        print(f"Found {len(hr_paths)} images in {self.hr_dir}")
        
        # Shuffle và split
        n = len(hr_paths)
        random.seed(self.seed)
        ids = list(range(n))
        random.shuffle(ids)
        
        s_tr, s_va, s_te = self.split
        n_tr = int(n * s_tr)
        n_va = int(n * s_va) 
        n_te = n - n_tr - n_va  # Remaining
        
        print(f"Split: Train={n_tr}, Val={n_va}, Test={n_te}")
        
        pick = lambda a, I: [a[i] for i in I]
        id_tr = ids[:n_tr]
        id_va = ids[n_tr:n_tr+n_va] 
        id_te = ids[n_tr+n_va:]

        # Augmentation chỉ cho training
        aug_train = build_aug(self.augment) if self.augment else None

        # Tạo datasets
        ds_tr = _SRDataset(pick(hr_paths, id_tr), self.scale, "train", 
                          aug=aug_train, patch_size=self.patch_size)
        ds_va = _SRDataset(pick(hr_paths, id_va), self.scale, "val", 
                          aug=None, patch_size=0)  # No cropping for val
        ds_te = _SRDataset(pick(hr_paths, id_te), self.scale, "test", 
                          aug=None, patch_size=0)  # No cropping for test

        # Tạo dataloaders
        mk = lambda ds, bs, sh: DataLoader(
            ds, batch_size=bs, shuffle=sh,
            num_workers=self.num_workers, pin_memory=True,
            drop_last=(sh and bs > 1)  # Drop last chỉ cho training
        )
        
        # Val/Test dùng batch_size=1 để xử lý ảnh có kích thước khác nhau
        return (mk(ds_tr, self.batch_size, True), 
                mk(ds_va, 1, False), 
                mk(ds_te, 1, False))

def build_sr_loader(**kwargs):
    """Build SR loader"""
    loader = SRLoader(**kwargs)
    return loader.make()