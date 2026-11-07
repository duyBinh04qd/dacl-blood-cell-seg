"""Chuẩn hóa bộ dữ liệu Blood Cell Segmentation (Depto et al., 2021) về một cấu trúc thống nhất.

Bộ dữ liệu gốc: ảnh 1600x1200, ảnh và mask ghép cặp theo tên
(ví dụ ``123_image.png`` và ``123_mask.jpg``, có thể nằm chung một thư mục
hoặc ở hai thư mục riêng). Mask lưu dạng JPG nên có nhiễu nén -> cần phân ngưỡng.

Kết quả sau khi chạy:
    <out>/images/<id>.png   ảnh RGB đã thu nhỏ
    <out>/masks/<id>.png    mask nhị phân 0/255 đã thu nhỏ

Cách dùng (Kaggle):
    python scripts/prepare_dataset.py --raw /kaggle/input/<ten-dataset> --out /kaggle/working/data/processed
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ---------------------------- CẤU HÌNH ----------------------------
# Từ khóa nhận diện file mask (trong tên file hoặc tên thư mục cha).
MASK_WORDS = ("mask", "gt", "label", "ground")
# Hậu tố bị bỏ khỏi tên file để lấy "khóa" ghép cặp ảnh <-> mask.
# Ví dụ: "123_image" và "123_mask" đều cho khóa "123".
SUFFIX_RE = re.compile(r"[ _\-]?(mask|masks|gt|label|labels|image|images|img|original)$", re.I)
# ------------------------------------------------------------------


def is_mask(path: Path) -> bool:
    name = path.stem.lower()
    parent = path.parent.name.lower()
    return any(w in name for w in MASK_WORDS) or any(w in parent for w in MASK_WORDS)


def key_of(path: Path) -> str:
    return SUFFIX_RE.sub("", path.stem).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="thư mục dữ liệu gốc (quét đệ quy)")
    ap.add_argument("--out", default="data/processed")
    # Kích thước lưu lại (rộng x cao). Giữ tỉ lệ 4:3 của ảnh gốc 1600x1200.
    # Thu nhỏ trước giúp DataLoader trên Kaggle nhanh hơn nhiều so với đọc ảnh 1600x1200
    # ở mỗi vòng lặp. Khi huấn luyện, ảnh sẽ còn được resize tiếp về --img_size.
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    # Ngưỡng nhị phân hóa mask: pixel > 127 là tế bào (1), còn lại là nền (0).
    # Cần thiết vì mask JPG có các giá trị trung gian do nén mất mát.
    ap.add_argument("--threshold", type=int, default=127)
    args = ap.parse_args()

    files = [p for p in Path(args.raw).rglob("*") if p.suffix.lower() in EXTS]
    images, masks = {}, {}
    for p in files:
        (masks if is_mask(p) else images)[key_of(p)] = p

    keys = sorted(set(images) & set(masks))
    print(f"Tìm thấy {len(images)} ảnh, {len(masks)} mask -> ghép được {len(keys)} cặp")
    if len(images) - len(keys):
        print("  Ảnh không có mask (ví dụ):", sorted(set(images) - set(masks))[:5])
    if len(masks) - len(keys):
        print("  Mask không có ảnh (ví dụ):", sorted(set(masks) - set(images))[:5])
    if not keys:
        raise SystemExit("Không ghép được cặp nào - hãy chạy inspect_dataset.py để xem cách đặt tên file.")

    out_img = Path(args.out) / "images"
    out_msk = Path(args.out) / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)

    size = (args.width, args.height)
    fg_ratio = []
    for i, k in enumerate(tqdm(keys, desc="Chuẩn hóa")):
        img = cv2.imread(str(images[k]), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(masks[k]), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            print("  Bỏ qua file lỗi:", k)
            continue
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        # Ảnh dùng INTER_AREA (mịn), mask dùng INTER_NEAREST để không sinh giá trị trung gian.
        msk = cv2.resize(msk, size, interpolation=cv2.INTER_NEAREST)
        msk = np.where(msk > args.threshold, 255, 0).astype(np.uint8)
        name = f"{i:04d}"  # đặt lại tên thống nhất 0000, 0001, ...
        cv2.imwrite(str(out_img / f"{name}.png"), img)
        cv2.imwrite(str(out_msk / f"{name}.png"), msk)
        fg_ratio.append((msk > 0).mean())

    print(f"Đã lưu {len(fg_ratio)} cặp vào {args.out}")
    print(f"Tỉ lệ pixel tế bào trung bình: {np.mean(fg_ratio):.3f} "
          f"(nếu ~0 hoặc ~1 thì kiểm tra lại ngưỡng/đảo màu mask)")


if __name__ == "__main__":
    main()
