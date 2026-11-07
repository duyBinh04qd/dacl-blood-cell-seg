"""Inspect the downloaded dataset: image sizes, file formats and the pixel values found in masks.

Usage:
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
    ap.add_argument("--sample", type=int, default=50, help="number of masks read to collect pixel values")
    args = ap.parse_args()

    imgs, masks = list_files(args.images), list_files(args.masks)
    print(f"Images: {len(imgs)} | Masks: {len(masks)}")
    print("Image formats:", Counter(p.suffix.lower() for p in imgs))
    print("Mask formats:", Counter(p.suffix.lower() for p in masks))

    sizes = Counter(Image.open(p).size for p in imgs)
    modes = Counter(Image.open(p).mode for p in imgs)
    print("Image sizes (W, H):", sizes.most_common(5))
    print("Image color modes:", modes)

    img_stems = {p.stem for p in imgs}
    mask_stems = {p.stem for p in masks}
    print("Images without a mask of the same name:", len(img_stems - mask_stems))

    values = Counter()
    for p in masks[: args.sample]:
        m = np.array(Image.open(p))
        if m.ndim == 3:
            m = m[..., 0]
        values.update(np.unique(m).tolist())
    print(f"Pixel values in the first {min(args.sample, len(masks))} masks:", sorted(values))
    print("=> Only {0, 255} (or {0, 1}) means binary masks: n_classes = 2.")


if __name__ == "__main__":
    main()
