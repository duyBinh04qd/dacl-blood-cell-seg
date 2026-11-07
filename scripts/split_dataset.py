"""Split the data 70/10/20 and create the 2% / 5% / 10% / 20% labeled lists.

Usage:
    python scripts/split_dataset.py --images data/processed/images --out data/splits
Output: train.txt, val.txt, test.txt, labeled_{2,5,10,20}.txt
(one file name without extension per line).
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
    # Keep the same seed across all experiments so every model uses the same split.
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
    for pct in (2, 5, 10, 20):
        k = int(len(train) * pct / 100)
        write(out / f"labeled_{pct}.txt", train[:k])  # the rest of train = unlabeled
        print(f"{pct}% labeled: {k} labeled / {len(train) - k} unlabeled images")
    print(f"train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == "__main__":
    main()
