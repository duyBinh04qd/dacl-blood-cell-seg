"""Binary segmentation metrics: DSC, Jaccard, HD95, ASD.

HD95 and ASD are in PIXELS (the dataset provides no physical pixel size)
and are computed at the evaluation resolution (--img_size).
"""
import numpy as np
from scipy import ndimage


def _surface(mask):
    """Boundary pixels of a binary mask."""
    eroded = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(2, 1))
    return mask ^ eroded


def _surface_distances(pred, gt):
    """Distance from every pred boundary pixel to the nearest gt boundary (and vice versa)."""
    sp, sg = _surface(pred), _surface(gt)
    dt_g = ndimage.distance_transform_edt(~sg)
    dt_p = ndimage.distance_transform_edt(~sp)
    return dt_g[sp], dt_p[sg]


def binary_metrics(pred, gt):
    """pred, gt: bool/0-1 arrays of shape (H, W). Returns a dict of metrics."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    psum, gsum = pred.sum(), gt.sum()
    union = np.logical_or(pred, gt).sum()

    # Empty-mask convention (as in MedPy/nnU-Net):
    # - both empty: perfect prediction -> DSC = 1, distance = 0
    # - only one empty: DSC = 0, distance = image diagonal (maximum penalty)
    if psum == 0 and gsum == 0:
        return {"dsc": 1.0, "jaccard": 1.0, "hd95": 0.0, "asd": 0.0}
    dsc = 2.0 * inter / (psum + gsum)
    jac = inter / union
    if psum == 0 or gsum == 0:
        diag = float(np.hypot(*pred.shape))
        return {"dsc": float(dsc), "jaccard": float(jac), "hd95": diag, "asd": diag}

    d_pg, d_gp = _surface_distances(pred, gt)
    all_d = np.concatenate([d_pg, d_gp])
    return {
        "dsc": float(dsc),
        "jaccard": float(jac),
        "hd95": float(np.percentile(all_d, 95)),
        "asd": float(all_d.mean()),
    }
