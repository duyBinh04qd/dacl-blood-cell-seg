"""Chia dữ liệu 70/10/20 và tạo danh sách ảnh có nhãn 5% / 10% / 20%.

Cách dùng:
    python scripts/split_dataset.py --images data/raw/images --out data/splits
Kết quả: train.txt, val.txt, test.txt, labeled_5.txt, labeled_10.txt, labeled_20.txt
(mỗi dòng là tên tệp không có phần mở rộng).
"""
import argparse
import random
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def write(path, items):
    path.write_text("\n".join(items) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="data/splits")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    names = sorted(p.stem for p in Path(args.images).iterdir() if p.suffix.lower() in EXTS)
    random.Random(args.seed).shuffle(names)

    n = len(names)
    n_train, n_val = round(0.7 * n), round(0.1 * n)
    train, val, test = names[:n_train], names[n_train:n_train + n_val], names[n_train + n_val:]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "train.txt", train)
    write(out / "val.txt", val)
    write(out / "test.txt", test)
    for pct in (5, 10, 20):
        k = int(len(train) * pct / 100)
        write(out / f"labeled_{pct}.txt", train[:k])  # phần còn lại của train = không nhãn
        print(f"{pct}% nhãn: {k} ảnh có nhãn / {len(train) - k} ảnh không nhãn")
    print(f"train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    main()
