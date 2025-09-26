# src/eval/postprocess.py
from typing import Optional, Callable, Dict
import torch
import torch.nn as nn

PostFunc = Callable[[torch.Tensor], torch.Tensor]

class Identity:
    def __init__(self, model: Optional[nn.Module] = None) -> None:
        self.model = model
    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) if self.model is not None else x

class TTAFlip:
    def __init__(self, model: nn.Module) -> None:
        self.model = model.eval()
    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        flips = [lambda z: z,
                 lambda z: torch.flip(z, [2]),
                 lambda z: torch.flip(z, [3]),
                 lambda z: torch.flip(z, [2,3])]
        for f in flips:
            y = self.model(f(x))
            outs.append(f(y))
        return torch.stack(outs, 0).mean(0).clamp(0, 1)

# --- Registry ---
_REGISTRY: Dict[str, Callable[[nn.Module], PostFunc]] = {
    "identity":  lambda model: Identity(model),
    "tta_flip":  lambda model: TTAFlip(model),
}

def build_postprocess(name: Optional[str], model: nn.Module) -> Optional[PostFunc]:
    if not name: return None
    if name not in _REGISTRY:
        raise ValueError(f"Unknown postprocess: {name}")
    return _REGISTRY[name](model)
