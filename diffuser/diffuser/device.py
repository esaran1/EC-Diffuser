"""Small, dependency-free device selection helpers."""

import torch


def get_available_device() -> torch.device:
    """Return the best available PyTorch device in CUDA, MPS, CPU order."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    return torch.device("cpu")
