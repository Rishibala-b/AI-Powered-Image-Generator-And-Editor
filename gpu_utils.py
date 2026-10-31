import os

import torch


def use_offload(device: str) -> bool:
    """True -> keep weights in RAM and move parts to the GPU only while they run.
    Automatic for GPUs under 6 GB (e.g. RTX 2050 4 GB).
    Force with LOW_VRAM=1 (on) or LOW_VRAM=0 (off) in .env."""
    if device != "cuda":
        return False
    flag = os.getenv("LOW_VRAM", "").strip()
    if flag in ("0", "1"):
        return flag == "1"
    return torch.cuda.get_device_properties(0).total_memory < 6 * 1024**3
