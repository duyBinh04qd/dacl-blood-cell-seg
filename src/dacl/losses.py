"""Các hàm mất mát của DACL (Wang et al., 2026).

Ký hiệu: logits có dạng (B, C, H, W). p là xác suất của CNN (UNeXt),
q là xác suất của Transformer (Swin-Unet).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class DiceLoss(nn.Module):
    """Dice loss nhiều lớp; target là chỉ số lớp (B, H, W)."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, logits, target):
        prob = torch.softmax(logits, dim=1)
        onehot = F.one_hot(target.long(), self.n_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (prob * onehot).sum(dims)
        union = prob.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + 1e-5) / (union + 1e-5)
        return 1 - dice.mean()


def ce_dice(logits, target, dice_fn):
    """H(.) + D(.) -- dùng cho cả L_sup và L_cps."""
    return F.cross_entropy(logits, target.long()) + dice_fn(logits, target)


def cps_loss(logits1, logits2, dice_fn):
    """Cross Pseudo Supervision (Eq. 1): nhãn giả one-hot, không truyền gradient."""
    pseudo1 = logits1.detach().argmax(dim=1)
    pseudo2 = logits2.detach().argmax(dim=1)
    l1 = ce_dice(logits1, pseudo2, dice_fn)
    l2 = ce_dice(logits2, pseudo1, dice_fn)
    return l1 + l2


def mixup(x_i, x_j, sigma: float = 0.5):
    return sigma * x_i + (1 - sigma) * x_j


def ict_loss(student_logits_mixed, teacher_logits_i, teacher_logits_j, sigma: float = 0.5):
    """Interpolation consistency: f2(Mix(xi,xj)) ~ Mix(f_ema(xi), f_ema(xj)).

    So sánh trên xác suất softmax, dùng MSE trung bình.
    """
    with torch.no_grad():
        target = mixup(torch.softmax(teacher_logits_i, 1), torch.softmax(teacher_logits_j, 1), sigma)
    return F.mse_loss(torch.softmax(student_logits_mixed, 1), target)


def _entropy(prob):
    """Entropy Shannon theo từng điểm ảnh -> (B, H, W)."""
    return -(prob * torch.log(prob + EPS)).sum(dim=1)


def dac_loss(logits_cnn, logits_tr):
    """Dual-Agreement Consistency: L_KL + L_ent.

    L_KL  = mean_u KL(p(u) || q(u))
    L_ent = mean_u [ H(p) + H(q) - H((p+q)/2) ]
    """
    p = torch.softmax(logits_cnn, dim=1)
    q = torch.softmax(logits_tr, dim=1)
    kl = (p * (torch.log(p + EPS) - torch.log(q + EPS))).sum(dim=1).mean()
    ent = (_entropy(p) + _entropy(q) - _entropy(0.5 * (p + q))).mean()
    return kl + ent


@torch.no_grad()
def update_ema(ema_model, model, alpha: float = 0.99, global_step: int = 0):
    """Cập nhật tham số giáo viên EMA."""
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(alpha).add_(p.detach(), alpha=1 - alpha)


def total_loss(sup1, sup2, cps, ict, dac, lam=2.0, tau=2.0, beta=10.0):
    """L_total = (L1_sup + L2_sup) + lambda*L_cps + tau*L_ict + beta*L_dac."""
    return (sup1 + sup2) + lam * cps + tau * ict + beta * dac


if __name__ == "__main__":
    # Kiểm tra nhanh với dữ liệu ngẫu nhiên
    torch.manual_seed(0)
    B, C, H, W = 2, 2, 64, 64
    dice = DiceLoss(C)
    a = torch.randn(B, C, H, W, requires_grad=True)
    b = torch.randn(B, C, H, W, requires_grad=True)
    y = torch.randint(0, C, (B, H, W))
    xi, xj = torch.randn(B, C, H, W), torch.randn(B, C, H, W)
    loss = total_loss(
        ce_dice(a, y, dice), ce_dice(b, y, dice),
        cps_loss(a, b, dice), ict_loss(b, xi, xj), dac_loss(a, b),
    )
    loss.backward()
    print("total loss =", float(loss), "| grad ok:", a.grad is not None and b.grad is not None)
