import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    """Fix random seeds for reproducibility (same seed -> same result, approximately on GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def poly_lr(base_lr: float, it: int, max_it: int, power: float = 0.9) -> float:
    """"Poly" learning-rate schedule: lr = base_lr * (1 - it/max_it)^0.9.
    The standard schedule in SSL4MIS and the CPS / CTCT / DACL papers."""
    return base_lr * (1 - it / max_it) ** power


def count_params(model) -> float:
    """Number of trainable parameters, in millions (M)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def infinite_loader(loader):
    """Iterate a DataLoader forever - training is driven by iterations, not epochs."""
    while True:
        for batch in loader:
            yield batch
