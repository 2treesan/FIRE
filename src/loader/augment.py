# src/loader/augment.py
from typing import List, Tuple, Protocol, Optional
from PIL import Image, ImageFilter, ImageEnhance
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

class FlipRot8:
    """Full dihedral-8 set: rotations + flips + (TRANSPOSE/TRANSVERSE)."""
    def __init__(self) -> None:
        self.ops = [
            "id", "fliph", "flipv",
            "rot90", "rot180", "rot270",
            "transpose", "transverse",
        ]
    def _apply(self, im: Image.Image, code: str) -> Image.Image:
        if code == "id": return im
        if code == "fliph":  return im.transpose(Image.FLIP_LEFT_RIGHT)
        if code == "flipv":  return im.transpose(Image.FLIP_TOP_BOTTOM)
        if code == "rot90":  return im.rotate(90, expand=False)
        if code == "rot180": return im.rotate(180, expand=False)
        if code == "rot270": return im.rotate(270, expand=False)
        if hasattr(Image, "TRANSPOSE") and code == "transpose":
            return im.transpose(Image.TRANSPOSE)
        if hasattr(Image, "TRANSVERSE") and code == "transverse":
            return im.transpose(Image.TRANSVERSE)
        return im
    def __call__(self, lr: Image.Image, hr: Image.Image):
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

def _bicubic():
    return getattr(getattr(Image, "Resampling", Image), "BICUBIC")

class LRDownUp:
    """
    Degrade LR by downscaling with a random factor and upscaling back to original size.
    This simulates extra aliasing/interpolation artifacts. HR is untouched.
    """
    def __init__(self,
                 p: float = 0.7,
                 scale_range: Tuple[float, float] = (0.5, 0.95),
                 resamples: Optional[List[int]] = None) -> None:
        self.p = float(p)
        self.scale_range = scale_range
        # resampling methods to choose when upscaling back
        if resamples is None:
            self.resamples = [
                getattr(getattr(Image, "Resampling", Image), "NEAREST"),
                getattr(getattr(Image, "Resampling", Image), "BILINEAR"),
                getattr(getattr(Image, "Resampling", Image), "BICUBIC"),
            ]
        else:
            self.resamples = resamples
    def __call__(self, lr: Image.Image, hr: Image.Image):
        if random.random() > self.p:
            return lr, hr
        w, h = lr.size
        r = random.uniform(*self.scale_range)
        w2, h2 = max(1, int(round(w * r))), max(1, int(round(h * r)))
        small = lr.resize((w2, h2), resample=_bicubic())
        back = small.resize((w, h), resample=random.choice(self.resamples))
        return back, hr

class LRUnsharp:
    """Apply unsharp mask on LR to simulate halos/oversharpening."""
    def __init__(self, p: float = 0.4,
                 radius: Tuple[float, float] = (0.5, 2.5),
                 percent: Tuple[int, int] = (50, 200),
                 threshold: Tuple[int, int] = (0, 5)) -> None:
        self.p = float(p)
        self.radius = radius
        self.percent = percent
        self.threshold = threshold
    def __call__(self, lr: Image.Image, hr: Image.Image):
        if random.random() > self.p:
            return lr, hr
        return (
            lr.filter(ImageFilter.UnsharpMask(
                radius=random.uniform(*self.radius),
                percent=random.randint(*self.percent),
                threshold=random.randint(*self.threshold)
            )),
            hr,
        )

class PairEnhance:
    """
    Apply the same color/contrast/brightness/sharpness adjustments to both LR and HR.
    Safe when LR and HR have different sizes (no spatial mixing).
    """
    def __init__(self,
                 p: float = 0.8,
                 brightness: Tuple[float, float] = (0.9, 1.1),
                 contrast: Tuple[float, float] = (0.9, 1.1),
                 saturation: Tuple[float, float] = (0.9, 1.1),
                 sharpness: Tuple[float, float] = (0.9, 1.1)) -> None:
        self.p = float(p)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.sharpness = sharpness

    def _rand(self, rng: Tuple[float, float]) -> float:
        if rng is None:
            return 1.0
        a, b = rng
        if a == 1.0 and b == 1.0:
            return 1.0
        return random.uniform(a, b)

    def _apply_factors(self, im: Image.Image, b: float, c: float, s: float, sh: float) -> Image.Image:
        out = ImageEnhance.Brightness(im).enhance(b)
        out = ImageEnhance.Contrast(out).enhance(c)
        out = ImageEnhance.Color(out).enhance(s)
        out = ImageEnhance.Sharpness(out).enhance(sh)
        return out

    def __call__(self, lr: Image.Image, hr: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() > self.p:
            return lr, hr
        b = self._rand(self.brightness)
        c = self._rand(self.contrast)
        s = self._rand(self.saturation)
        sh = self._rand(self.sharpness)
        return self._apply_factors(lr, b, c, s, sh), self._apply_factors(hr, b, c, s, sh)

class PairGamma:
    """Apply the same gamma correction to both LR and HR."""
    def __init__(self, p: float = 0.5, gamma: Tuple[float, float] = (0.8, 1.2)) -> None:
        self.p = float(p)
        self.gamma = gamma
    def __call__(self, lr: Image.Image, hr: Image.Image):
        if random.random() > self.p:
            return lr, hr
        g = float(random.uniform(*self.gamma))
        def apply(im: Image.Image) -> Image.Image:
            arr = np.asarray(im).astype(np.float32) / 255.0
            out = np.power(np.clip(arr, 0.0, 1.0), g)
            return Image.fromarray(np.clip(out * 255.0, 0, 255).astype(np.uint8))
        return apply(lr), apply(hr)

class PairHue:
    """Shift hue by the same amount for both LR and HR (in HSV space)."""
    def __init__(self, p: float = 0.5, max_deg: float = 10.0) -> None:
        self.p = float(p)
        self.max_deg = float(max_deg)
    def __call__(self, lr: Image.Image, hr: Image.Image):
        if random.random() > self.p:
            return lr, hr
        # PIL HSV uses H in [0,255]; map degrees to that range
        shift = int(round((random.uniform(-self.max_deg, self.max_deg) / 360.0) * 255.0))
        def apply(im: Image.Image) -> Image.Image:
            hsv = np.array(im.convert("HSV"), dtype=np.uint8)
            h = hsv[:, :, 0].astype(np.int16)
            h = (h + shift) % 256
            hsv[:, :, 0] = h.astype(np.uint8)
            return Image.fromarray(hsv, mode="HSV").convert("RGB")
        return apply(lr), apply(hr)

class PairRandomGrayscale:
    """Randomly convert to grayscale and back for both LR and HR (paired)."""
    def __init__(self, p: float = 0.1) -> None:
        self.p = float(p)
    def __call__(self, lr: Image.Image, hr: Image.Image):
        if random.random() > self.p:
            return lr, hr
        return lr.convert("L").convert("RGB"), hr.convert("L").convert("RGB")

class PairCutBlur:
    """
    Size-aware CutBlur: swap/copy a rectangle between LR and HR by mapping with the inferred scale.
    This version is safe when LR and HR have different sizes.
    """
    def __init__(self, p: float = 0.5, frac: Tuple[float, float] = (0.125, 0.5)) -> None:
        self.p = float(p)
        self.frac = frac
    def __call__(self, lr: Image.Image, hr: Image.Image):
        if random.random() > self.p:
            return lr, hr
        lw, lh = lr.size; hw, hh = hr.size
        if lw == 0 or lh == 0 or hw == 0 or hh == 0:
            return lr, hr
        # infer integer scale; if not divisible, skip
        if hw % lw != 0 or hh % lh != 0:
            return lr, hr
        s_x, s_y = hw // lw, hh // lh
        if s_x != s_y:
            return lr, hr
        s = s_x
        # pick HR-region size as multiples of scale
        min_w = max(s, int(hw * self.frac[0]) // s * s)
        max_w = max(min_w, int(hw * self.frac[1]) // s * s)
        min_h = max(s, int(hh * self.frac[0]) // s * s)
        max_h = max(min_h, int(hh * self.frac[1]) // s * s)
        cw = random.randrange(min_w, max_w + 1, s)
        ch = random.randrange(min_h, max_h + 1, s)
        if cw <= 0 or ch <= 0:
            return lr, hr
        x = random.randint(0, hw - cw)
        y = random.randint(0, hh - ch)
        lr_box = (x // s, y // s, (x + cw) // s, (y + ch) // s)
        hr_box = (x, y, x + cw, y + ch)
        lr2, hr2 = lr.copy(), hr.copy()
        if random.random() < 0.5:
            # HR -> LR (downsample HR patch and paste to LR)
            patch_hr = hr.crop(hr_box)
            patch_lr = patch_hr.resize((cw // s, ch // s), resample=_bicubic())
            lr2.paste(patch_lr, lr_box)
        else:
            # LR -> HR (upsample LR patch and paste to HR)
            patch_lr = lr.crop(lr_box)
            patch_hr = patch_lr.resize((cw, ch), resample=_bicubic())
            hr2.paste(patch_hr, hr_box)
        return lr2, hr2

# PRESET REGISTRY 
PRESETS = {
    # geometric
    "fliprot":  FlipRot(),             # 6-way (flips + 90/180/270)
    "fliprot8": FlipRot8(),            # full dihedral-8

    # photometric (paired, size-agnostic)
    "pair_enhance": PairEnhance(),     # brightness/contrast/saturation/sharpness
    "pair_gamma":   PairGamma(),
    "pair_hue":     PairHue(),
    "pair_gray":    PairRandomGrayscale(),
    "pair_cutblur": PairCutBlur(),     # spatial swap with proper LR<->HR mapping

    # degradation (LR-only)
    "degrade_std": DegradeStd(),       # gaussian blur + noise + jpeg
    "lr_downup":   LRDownUp(),         # extra down-up resize
    "lr_unsharp":  LRUnsharp(),        # oversharpen halos
}

def build_pipeline(names: List[str]) -> Compose:
    ops: List[Aug] = []
    for n in names:
        if n not in PRESETS:
            raise ValueError(f"Unknown augment: {n}")
        ops.append(PRESETS[n])
    return Compose(ops)
