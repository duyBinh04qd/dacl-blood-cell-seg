import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    """Cố định seed để kết quả có thể tái lập (cùng seed -> cùng kết quả, gần đúng trên GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def poly_lr(base_lr: float, it: int, max_it: int, power: float = 0.9) -> float:
    """Lịch learning rate dạng "poly": lr = base_lr * (1 - it/max_it)^0.9.
    Đây là lịch chuẩn trong SSL4MIS / các bài CPS, CTCT, DACL."""
    return base_lr * (1 - it / max_it) ** power


def count_params(model) -> float:
    """Số tham số huấn luyện được, tính theo triệu (M)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def infinite_loader(loader):
    """Lặp DataLoader vô hạn - vì ta huấn luyện theo số vòng lặp chứ không theo epoch."""
    while True:
        for batch in loader:
            yield batch
