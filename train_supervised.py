"""FULLY SUPERVISED (FS) training for UNeXt or Swin-Unet.

These are the reference rows of the results table:
  - FS with 100% labels  -> upper bound
  - FS with 5/10/20% labels (labeled images only, unlabeled ones discarded) -> SSL lower bound

Examples:
  python train_supervised.py --model unext     --labeled_ratio 100
  python train_supervised.py --model swin_unet --labeled_ratio 10 --pretrained imagenet
  python train_supervised.py --model unext --eval_only --ckpt runs/unext_100p_s2026/best.pth
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dacl.losses import DiceLoss, ce_dice
from src.data.dataset import BloodCellDataset, read_list
from src.data.transforms import eval_transform, train_transform
from src.models import build_model
from src.utils.common import count_params, infinite_loader, poly_lr, set_seed
from src.utils.metrics import binary_metrics


def get_args():
    ap = argparse.ArgumentParser(description="Fully supervised training of UNeXt / Swin-Unet")

    # ======================= MODEL =======================
    # unext     : lightweight CNN (~1.5M params) - the model deployed by DACL.
    # swin_unet : Transformer (~27M params) - the "teacher" in DACL.
    ap.add_argument("--model", choices=["unext", "swin_unet"], required=True)
    # Number of segmentation classes: 2 = background (0) + cell (1). Dataset masks are binary.
    ap.add_argument("--num_classes", type=int, default=2)
    # UNeXt channel configuration: "dacl" (32,64,128,160,256; 1.58M) or "original" (16,32,...; 1.47M).
    # Use the SAME configuration for FS-UNeXt and the UNeXt inside DACL for a fair comparison.
    ap.add_argument("--unext_variant", choices=["dacl", "original"], default="dacl")
    # Swin-Unet initialization:
    #   "imagenet"   -> download Microsoft's Swin-T (Kaggle: turn Internet ON in Settings)
    #   "<path.pth>" -> local file (with Internet OFF, upload the file as a Kaggle Dataset)
    #   ""           -> random init (NOT recommended - DACL uses the pretrained model)
    # UNeXt is always trained from scratch; this option is ignored for it.
    ap.add_argument("--pretrained", default="imagenet")

    # ======================= DATA =======================
    # Folder created by prepare_dataset.py (contains images/ and masks/).
    ap.add_argument("--data_dir", default="data/processed")
    # Folder created by split_dataset.py (train.txt, val.txt, test.txt, labeled_X.txt).
    ap.add_argument("--split_dir", default="data/splits")
    # Percentage of labeled images used for training: 100 = the whole train.txt,
    # 2/5/10/20 = read labeled_X.txt (identical to the SSL experiments).
    ap.add_argument("--labeled_ratio", type=int, default=100)
    # Network input size. 224 matches Swin-Unet (window 7, ImageNet weights at 224).
    # Use the SAME size for both models for a fair comparison.
    ap.add_argument("--img_size", type=int, default=224)

    # ======================= TRAINING =======================
    # Train for a fixed NUMBER OF ITERATIONS rather than epochs: every label ratio gets the
    # same number of updates (with 46 images an epoch is only a few steps). 30k is common in SSL4MIS.
    # Tip: run --max_iters 500 first to sanity-check the pipeline.
    ap.add_argument("--max_iters", type=int, default=30000)
    # Batch size. 8 fits a 16GB T4/P100 for both models at 224x224.
    # Reduce to 4 if Swin-Unet raises "CUDA out of memory".
    ap.add_argument("--batch_size", type=int, default=8)
    # SGD as in the DACL paper: lr 0.01, momentum 0.9, weight decay 1e-4,
    # decayed with the poly schedule lr * (1 - iter/max_iters)^0.9.
    ap.add_argument("--base_lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    # Mixed precision (FP16): ~1.5-2x faster and less memory on Kaggle GPUs.
    ap.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    # Data-loading workers. Kaggle has 4 CPUs -> 2-4 is reasonable.
    ap.add_argument("--num_workers", type=int, default=4)
    # Random seed - rerun with other seeds (e.g. 2026, 2027, 2028) to report mean ± std.
    ap.add_argument("--seed", type=int, default=2026)

    # ======================= EVALUATION & SAVING =======================
    # Every val_interval iterations evaluate on the validation set and save best.pth when
    # the validation Dice improves (as in the paper: keep the best validation weights).
    ap.add_argument("--val_interval", type=int, default=500)
    ap.add_argument("--log_interval", type=int, default=50)
    # Output folder. On Kaggle only /kaggle/working is kept after the run.
    # Default: runs/<model>_<ratio>p_s<seed>
    ap.add_argument("--out_dir", default="")
    # Resume from last.pth (e.g. after a Kaggle session was interrupted).
    ap.add_argument("--resume", action="store_true")
    # Only evaluate a given checkpoint on the test set, no training.
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--ckpt", default="")
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, full_metrics=False):
    """Return (mean Dice, list of per-image metrics)."""
    model.eval()
    rows = []
    for batch in loader:
        img = batch["image"].to(device)
        pred = model(img).argmax(1).cpu().numpy()
        gt = batch["mask"].numpy()
        for p, g, n in zip(pred, gt, batch["name"]):
            if full_metrics:
                m = binary_metrics(p == 1, g == 1)
            else:  # validation: Dice only, for speed
                inter = np.logical_and(p == 1, g == 1).sum()
                s = (p == 1).sum() + (g == 1).sum()
                m = {"dsc": 1.0 if s == 0 else 2.0 * inter / s}
            m["name"] = n
            rows.append(m)
    model.train()
    return float(np.mean([r["dsc"] for r in rows])), rows


def run_test(model, args, device, out_dir):
    names = read_list(Path(args.split_dir) / "test.txt")
    ds = BloodCellDataset(args.data_dir, names, eval_transform(args.img_size))
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=args.num_workers)
    _, rows = evaluate(model, loader, device, full_metrics=True)

    keys = ["dsc", "jaccard", "hd95", "asd"]
    with open(out_dir / "test_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name"] + keys)
        w.writeheader()
        w.writerows(rows)
    # DSC and Jaccard x100 (%) to match the paper's tables; HD95 and ASD in pixels.
    summary = {}
    for k in keys:
        v = np.array([r[k] for r in rows]) * (100 if k in ("dsc", "jaccard") else 1)
        summary[k] = {"mean": float(v.mean()), "std": float(v.std())}
    (out_dir / "test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n===== TEST RESULTS =====")
    for k in keys:
        print(f"{k.upper():8s}: {summary[k]['mean']:.2f} ({summary[k]['std']:.2f})")
    return summary


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    out_dir = Path(args.out_dir or f"runs/{args.model}_{args.labeled_ratio}p_s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model, args.num_classes, args.img_size,
                        pretrained=args.pretrained if (args.model == "swin_unet" and not args.eval_only) else "",
                        unext_variant=args.unext_variant)
    model.to(device)
    print(f"Model {args.model}: {count_params(model):.2f}M params | device: {device}")

    if args.eval_only:
        ckpt = torch.load(args.ckpt or out_dir / "best.pth", map_location=device)
        model.load_state_dict(ckpt["model"])
        run_test(model, args, device, out_dir)
        return

    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # ---- Data ----
    split = Path(args.split_dir)
    train_names = read_list(split / ("train.txt" if args.labeled_ratio == 100
                                     else f"labeled_{args.labeled_ratio}.txt"))
    val_names = read_list(split / "val.txt")
    print(f"Labeled training images: {len(train_names)} | validation: {len(val_names)}")
    train_ds = BloodCellDataset(args.data_dir, train_names, train_transform(args.img_size))
    val_ds = BloodCellDataset(args.data_dir, val_names, eval_transform(args.img_size))
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=min(args.batch_size, len(train_ds)), shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda",
                              drop_last=True, generator=g, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=args.num_workers)

    # ---- Optimization ----
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    dice_loss = DiceLoss(args.num_classes)

    start_it, best_dice = 0, -1.0
    if args.resume and (out_dir / "last.pth").exists():
        ck = torch.load(out_dir / "last.pth", map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_it, best_dice = ck["iter"], ck["best_dice"]
        print(f"Resuming from iteration {start_it} (best Dice = {best_dice:.4f})")

    log_f = open(out_dir / "log.csv", "a", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    if start_it == 0:
        log.writerow(["iter", "lr", "loss", "val_dice"])

    # ---- Training loop ----
    model.train()
    data_iter = infinite_loader(train_loader)
    t0, running = time.time(), 0.0
    for it in range(start_it, args.max_iters):
        batch = next(data_iter)
        img = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        lr = poly_lr(args.base_lr, it, args.max_iters)
        for gparam in optimizer.param_groups:
            gparam["lr"] = lr

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(img)
        # Loss in FP32 for stability. L_sup = Cross-Entropy + Dice (as in DACL).
        loss = ce_dice(logits.float(), mask, dice_loss)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()

        step = it + 1
        if step % args.log_interval == 0:
            speed = args.log_interval / (time.time() - t0)
            print(f"[{step:6d}/{args.max_iters}] loss {running / args.log_interval:.4f} "
                  f"| lr {lr:.5f} | {speed:.1f} it/s")
            log.writerow([step, f"{lr:.6f}", f"{running / args.log_interval:.5f}", ""])
            running, t0 = 0.0, time.time()

        if step % args.val_interval == 0 or step == args.max_iters:
            val_dice, _ = evaluate(model, val_loader, device)
            log.writerow([step, "", "", f"{val_dice:.5f}"])
            log_f.flush()
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({"model": model.state_dict(), "iter": step, "val_dice": val_dice},
                           out_dir / "best.pth")
            print(f"  -> val Dice {val_dice:.4f} | best {best_dice:.4f}")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "iter": step, "best_dice": best_dice},
                       out_dir / "last.pth")
    log_f.close()

    # ---- Test-set evaluation with the best checkpoint ----
    model.load_state_dict(torch.load(out_dir / "best.pth", map_location=device)["model"])
    run_test(model, args, device, out_dir)


if __name__ == "__main__":
    main()
