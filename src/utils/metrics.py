"""Các chỉ số đánh giá phân đoạn nhị phân: DSC, Jaccard, HD95, ASD.

HD95 và ASD tính theo ĐƠN VỊ PIXEL (bộ dữ liệu không có kích thước vật lý của pixel)
và tính trên ảnh ở độ phân giải đánh giá (--img_size).
"""
import numpy as np
from scipy import ndimage


def _surface(mask):
    """Tập điểm biên của mask nhị phân."""
    eroded = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(2, 1))
    return mask ^ eroded


def _surface_distances(pred, gt):
    """Khoảng cách từ mỗi điểm biên của pred tới biên gần nhất của gt (và ngược lại)."""
    sp, sg = _surface(pred), _surface(gt)
    dt_g = ndimage.distance_transform_edt(~sg)
    dt_p = ndimage.distance_transform_edt(~sp)
    return dt_g[sp], dt_p[sg]


def binary_metrics(pred, gt):
    """pred, gt: mảng bool/0-1 kích thước (H, W). Trả về dict các chỉ số."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    psum, gsum = pred.sum(), gt.sum()
    union = np.logical_or(pred, gt).sum()

    # Quy ước khi mask rỗng (giống MedPy/nnU-Net):
    # - cả hai rỗng: dự đoán đúng hoàn toàn -> DSC = 1, khoảng cách = 0
    # - chỉ một bên rỗng: DSC = 0, khoảng cách = đường chéo ảnh (phạt tối đa)
    if psum == 0 and gsum == 0:
        return {"dsc": 1.0, "jaccard": 1.0, "hd95": 0.0, "asd": 0.0}
    dsc = 2.0 * inter / (psum + gsum)
    jac = inter / union
    if psum == 0 or gsum == 0:
        diag = float(np.hypot(*pred.shape))
        return {"dsc": dsc, "jaccard": jac, "hd95": diag, "asd": diag}

    d_pg, d_gp = _surface_distances(pred, gt)
    all_d = np.concatenate([d_pg, d_gp])
    return {
        "dsc": float(dsc),
        "jaccard": float(jac),
        "hd95": float(np.percentile(all_d, 95)),
        "asd": float(all_d.mean()),
    }
