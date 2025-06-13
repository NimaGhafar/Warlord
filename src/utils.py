import os
import torch

def _select_device() -> torch.device:
    """
    Kies automatisch het beste beschikbare device.

    Prioriteit
    ----------
    1. FORCE_DEVICE  –  omgevingsvariabele ('cuda' | 'mps' | 'cpu')
    2. CUDA          –  NVIDIA-GPU (torch.cuda)
    3. MPS           –  Apple-GPU   (torch.backends.mps)
    4. CPU           –  fallback
    """
    forced = os.getenv("FORCE_DEVICE")    
    if forced in {"cuda", "mps", "cpu"}:
        return torch.device(forced)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

device = _select_device()