"""Kiểm tra bộ dữ liệu sau khi tải: kích thước ảnh, định dạng, giá trị trong mask.

Cách dùng:
    python scripts/inspect_dataset.py --images data/raw/images --masks data/raw/masks
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_files(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--sample", type=int, default=50, help="số mask đọc để thống kê giá trị")
    args = ap.parse_args()

    imgs, masks = list_files(args.images), list_files(args.masks)
    print(f"Số ảnh: {len(imgs)} | Số mask: {len(masks)}")
    print("Định dạng ảnh:", Counter(p.suffix.lower() for p in imgs))
    print("Định dạng mask:", Counter(p.suffix.lower() for p in masks))

    sizes = Counter(Image.open(p).size for p in imgs)
    modes = Counter(Image.open(p).mode for p in imgs)
    print("Kích thước ảnh (W, H):", sizes.most_common(5))
    print("Chế độ màu ảnh:", modes)

    img_stems = {p.stem for p in imgs}
    mask_stems = {p.stem for p in masks}
    print("Ảnh không có mask cùng tên:", len(img_stems - mask_stems))

    values = Counter()
    for p in masks[: args.sample]:
        m = np.array(Image.open(p))
        if m.ndim == 3:
            m = m[..., 0]
        values.update(np.unique(m).tolist())
    print(f"Giá trị điểm ảnh trong {min(args.sample, len(masks))} mask đầu:", sorted(values))
    print("=> Nếu chỉ có {0, 255} (hoặc {0, 1}) thì đây là mask nhị phân: n_classes = 2.")


if __name__ == "__main__":
    main()
