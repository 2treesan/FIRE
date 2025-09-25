# src/loader/augment.py
from typing import List, Tuple, Protocol
from PIL import Image, ImageFilter
import random, io, numpy as np

class Aug(Protocol):
    def __call__(self, lr: Image.Image, hr: Image.Image) -> Tuple[Image.Image, Image.Image]: ...

class Compose:
    def __init__(self, ops: List[Aug]) -> None:
        self.ops = ops
    def __call__(self, lr: Image.Image, hr: Image.Image):
        for op in self.ops:
            lr, hr = op(lr, hr)
        return lr, hr

class Crop:
    def __init__(self, patch_size: int) -> None:
        self.ps = patch_size
    def __call__(self, lr, hr):
        if self.ps <= 0 or self.ps >= min(*lr.size): return lr, hr
        w, h = lr.size
        x = random.randint(0, w - self.ps); y = random.randint(0, h - self.ps)
        box = (x, y, x + self.ps, y + self.ps)
        return lr.crop(box), hr.crop(box)

class CenterCrop:
    def __init__(self, patch_size: int) -> None:
        self.ps = patch_size
    def __call__(self, lr, hr):
        if self.ps <= 0 or self.ps >= min(*lr.size): return lr, hr
        w,h = lr.size; x=(w-self.ps)//2; y=(h-self.ps)//2
        box=(x,y,x+self.ps,y+self.ps)
        return lr.crop(box), hr.crop(box)

class FlipRot:
    def __init__(self) -> None:
        # dùng mã hoá op thay vì lambda để picklable
        self.ops = ["id", "fliph", "flipv", "rot90", "rot180", "rot270"]
    def _apply(self, im, code: str):
        if code == "id":     return im
        if code == "fliph":  return im.transpose(Image.FLIP_LEFT_RIGHT)
        if code == "flipv":  return im.transpose(Image.FLIP_TOP_BOTTOM)
        if code == "rot90":  return im.rotate(90, expand=False)
        if code == "rot180": return im.rotate(180, expand=False)
        if code == "rot270": return im.rotate(270, expand=False)
        return im
    def __call__(self, lr, hr):
        op = random.choice(self.ops)
        return self._apply(lr, op), self._apply(hr, op)

class DegradeStd:
    def __init__(self) -> None:
        self.blur = (0.6, 2.4); self.p_noise = 0.4; self.nstd = (1,6)
        self.p_jpeg = 0.5; self.q = (40,95)
    def __call__(self, lr, hr):
        lr2 = lr.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.blur)))
        if random.random() < self.p_noise:
            arr = np.asarray(lr2).astype(np.float32)
            arr += np.random.normal(0, random.uniform(*self.nstd), arr.shape)
            lr2 = Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
        if random.random() < self.p_jpeg:
            buf = io.BytesIO(); lr2.save(buf, format="JPEG", quality=random.randint(*self.q)); buf.seek(0)
            lr2 = Image.open(buf).convert("RGB")
        return lr2, hr

class CutBlur:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
    def __call__(self, lr, hr):
        if random.random() > self.p: return lr, hr
        w, h = lr.size
        cw, ch = random.randint(w//8, w//2), random.randint(h//8, h//2)
        cx, cy = random.randint(0, w-cw), random.randint(0, h-ch)
        la, ha = np.array(lr), np.array(hr)
        if random.random() < 0.5:
            la[cy:cy+ch, cx:cx+cw] = ha[cy:cy+ch, cx:cx+cw]
        else:
            ha[cy:cy+ch, cx:cx+cw] = la[cy:cy+ch, cx:cx+cw]
        return Image.fromarray(la), Image.fromarray(ha)

# PRESET REGISTRY 
PRESETS = {
    "nocrop":   Crop(0),
    "crop64":   Crop(64),
    "crop96":   Crop(96),
    "crop128":  Crop(128),
    "crop192":  Crop(192),
    "centercrop128": CenterCrop(128),
    "fliprot":  FlipRot(),
    "degrade_std": DegradeStd(),
    "cutblur":  CutBlur(0.5),
}

def build_pipeline(names: List[str]) -> Compose:
    ops: List[Aug] = []
    for n in names:
        if n not in PRESETS:
            raise ValueError(f"Unknown augment: {n}")
        ops.append(PRESETS[n])
    return Compose(ops)
