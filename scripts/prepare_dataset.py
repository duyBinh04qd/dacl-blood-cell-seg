"""Normalize the Blood Cell Segmentation dataset (Depto et al., 2021) into a single layout.

Expected layout (as on Kaggle):
    bloodcell_dataset/
      ├── Original/   microscopy images
      └── Mask/       ground-truth masks, paired with images by file name
Extensions and suffixes such as ``_mask`` are ignored when matching names.
Masks may be stored as JPG (compression noise), so they are thresholded.

Output:
    <out>/images/<id>.png   resized RGB image
    <out>/masks/<id>.png    resized binary mask (0/255)

Usage (Kaggle):
    python scripts/prepare_dataset.py --raw /kaggle/input/bloodcell/bloodcell_dataset --out /kaggle/working/data/processed
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Suffixes stripped from file names to obtain the image <-> mask pairing key.
# Example: "123_image" and "123_mask" both give the key "123".
SUFFIX_RE = re.compile(r"[ _\-]?(mask|masks|gt|label|labels|image|images|img|original)$", re.I)


def index_folder(folder: Path) -> dict:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)
    return {SUFFIX_RE.sub("", p.stem).strip().lower(): p for p in files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder containing the image and mask sub-folders")
    ap.add_argument("--image_subdir", default="Original")
    ap.add_argument("--mask_subdir", default="Mask")
    ap.add_argument("--out", default="data/processed")
    # Saved size (width x height). Keeps the 4:3 aspect ratio of the 1600x1200 originals.
    # Resizing once up front makes the Kaggle DataLoader much faster than decoding
    # 1600x1200 images at every iteration. Images are resized again to --img_size for training.
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    # Mask binarization threshold: pixel > 127 is cell (1), everything else is background (0).
    # Needed because lossy JPG compression creates intermediate values.
    ap.add_argument("--threshold", type=int, default=127)
    args = ap.parse_args()

    raw = Path(args.raw)
    images = index_folder(raw / args.image_subdir)
    masks = index_folder(raw / args.mask_subdir)

    keys = sorted(set(images) & set(masks))
    print(f"Found {len(images)} images, {len(masks)} masks -> {len(keys)} pairs")
    if len(images) - len(keys):
        print("  Images without a mask (examples):", [images[k].name for k in sorted(set(images) - set(masks))[:5]])
    if len(masks) - len(keys):
        print("  Masks without an image (examples):", [masks[k].name for k in sorted(set(masks) - set(images))[:5]])
    if not keys:
        raise SystemExit("No pairs found - the file names in the two folders do not match.")
    for k in keys[:3]:
        print(f"  pair: {images[k].name}  <->  {masks[k].name}")

    out_img = Path(args.out) / "images"
    out_msk = Path(args.out) / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)

    size = (args.width, args.height)
    fg_ratio = []
    for i, k in enumerate(tqdm(keys, desc="Preparing")):
        img = cv2.imread(str(images[k]), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(masks[k]), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            print("  Skipping unreadable file:", k)
            continue
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        # INTER_AREA (smooth) for images, INTER_NEAREST for masks so no intermediate values appear.
        msk = cv2.resize(msk, size, interpolation=cv2.INTER_NEAREST)
        msk = np.where(msk > args.threshold, 255, 0).astype(np.uint8)
        name = f"{i:04d}"  # unified names 0000, 0001, ...
        cv2.imwrite(str(out_img / f"{name}.png"), img)
        cv2.imwrite(str(out_msk / f"{name}.png"), msk)
        fg_ratio.append((msk > 0).mean())

    print(f"Saved {len(fg_ratio)} pairs to {args.out}")
    print(f"Mean cell-pixel ratio: {np.mean(fg_ratio):.3f} "
          f"(if ~0 or ~1, check the threshold / mask inversion)")


if __name__ == "__main__":
    main()
