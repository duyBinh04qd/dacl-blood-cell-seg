"""Huấn luyện CÓ GIÁM SÁT (Fully Supervised - FS) cho UNeXt hoặc Swin-Unet.

Đây là các mốc so sánh trong bảng kết quả:
  - FS với 100% nhãn  -> giới hạn trên
  - FS với 5/10/20% nhãn (chỉ dùng phần ảnh có nhãn, bỏ ảnh không nhãn) -> giới hạn dưới của SSL

Ví dụ:
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
    ap = argparse.ArgumentParser(description="Huấn luyện có giám sát UNeXt / Swin-Unet")

    # ======================= MÔ HÌNH =======================
    # unext     : CNN nhẹ (~1.47M tham số) - mô hình sẽ triển khai trong DACL.
    # swin_unet : Transformer (~27M tham số) - "người thầy" trong DACL.
    ap.add_argument("--model", choices=["unext", "swin_unet"], required=True)
    # Số lớp phân đoạn: 2 = nền (0) + tế bào (1). Mask của bộ dữ liệu là nhị phân.
    ap.add_argument("--num_classes", type=int, default=2)
    # Cấu hình số kênh UNeXt: "dacl" (32,64,128,160,256; 1.58M) hoặc "original" (16,32,...; 1.47M).
    # Phải dùng CÙNG một cấu hình cho FS UNeXt và UNeXt trong DACL để so sánh công bằng.
    ap.add_argument("--unext_variant", choices=["dacl", "original"], default="dacl")
    # Trọng số khởi tạo cho Swin-Unet:
    #   "imagenet"   -> tự tải Swin-T của Microsoft (Kaggle cần BẬT Internet trong Settings)
    #   "<path.pth>" -> file đã tải sẵn (khi tắt Internet, upload file thành Kaggle Dataset)
    #   ""           -> khởi tạo ngẫu nhiên (KHÔNG khuyến nghị - DACL dùng bản tiền huấn luyện)
    # UNeXt luôn huấn luyện từ đầu, tham số này bị bỏ qua.
    ap.add_argument("--pretrained", default="imagenet")

    # ======================= DỮ LIỆU =======================
    # Thư mục do prepare_dataset.py tạo (chứa images/ và masks/).
    ap.add_argument("--data_dir", default="data/processed")
    # Thư mục do split_dataset.py tạo (train.txt, val.txt, test.txt, labeled_X.txt).
    ap.add_argument("--split_dir", default="data/splits")
    # Phần trăm ảnh có nhãn dùng để huấn luyện: 100 = toàn bộ train.txt,
    # 5/10/20 = đọc labeled_5.txt / labeled_10.txt / labeled_20.txt (khớp với các thí nghiệm SSL).
    ap.add_argument("--labeled_ratio", type=int, default=100)
    # Kích thước ảnh vào mạng. 224 tương thích Swin-Unet (window 7, trọng số ImageNet 224).
    # Dùng CÙNG kích thước cho cả hai mô hình để so sánh công bằng.
    ap.add_argument("--img_size", type=int, default=224)

    # ======================= HUẤN LUYỆN =======================
    # Huấn luyện theo SỐ VÒNG LẶP thay vì epoch: số bước cập nhật như nhau ở mọi mức nhãn
    # (với 46 ảnh, 1 epoch chỉ vài bước). 30k vòng là mức phổ biến trong SSL4MIS.
    # Mẹo: chạy thử --max_iters 500 trước để kiểm tra pipeline.
    ap.add_argument("--max_iters", type=int, default=30000)
    # Kích thước batch. 8 vừa GPU T4/P100 (16GB) cho cả hai mô hình ở 224x224.
    # Nếu báo lỗi "CUDA out of memory" với Swin-Unet thì giảm xuống 4.
    ap.add_argument("--batch_size", type=int, default=8)
    # Optimizer SGD theo bài báo DACL: lr 0.01, momentum 0.9, weight decay 1e-4,
    # lr giảm dần theo lịch poly: lr * (1 - iter/max_iters)^0.9.
    ap.add_argument("--base_lr", type=float, default=0.01)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    # Mixed precision (FP16): nhanh hơn ~1.5-2x và tốn ít bộ nhớ hơn trên GPU Kaggle.
    ap.add_argument("--no_amp", action="store_true", help="tắt mixed precision")
    # Số tiến trình đọc dữ liệu. Kaggle có 4 CPU -> 2-4 là hợp lý.
    ap.add_argument("--num_workers", type=int, default=4)
    # Seed ngẫu nhiên - chạy lại với seed khác (vd 2026, 2027, 2028) để báo cáo mean ± std.
    ap.add_argument("--seed", type=int, default=2026)

    # ======================= ĐÁNH GIÁ & LƯU =======================
    # Cứ mỗi val_interval vòng lặp thì đánh giá trên tập validation và lưu best.pth
    # nếu Dice validation tốt hơn (giống bài báo: lưu trọng số tốt nhất trên validation).
    ap.add_argument("--val_interval", type=int, default=500)
    ap.add_argument("--log_interval", type=int, default=50)
    # Thư mục kết quả. Trên Kaggle, chỉ /kaggle/working được giữ lại sau khi chạy xong.
    # Mặc định: runs/<model>_<ratio>p_s<seed>
    ap.add_argument("--out_dir", default="")
    # Tiếp tục huấn luyện từ last.pth (khi phiên Kaggle bị ngắt).
    ap.add_argument("--resume", action="store_true")
    # Chỉ đánh giá trên tập test với checkpoint cho sẵn, không huấn luyện.
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--ckpt", default="")
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, full_metrics=False):
    """Trả về (Dice trung bình, danh sách chỉ số theo từng ảnh)."""
    model.eval()
    rows = []
    for batch in loader:
        img = batch["image"].to(device)
        pred = model(img).argmax(1).cpu().numpy()
        gt = batch["mask"].numpy()
        for p, g, n in zip(pred, gt, batch["name"]):
            if full_metrics:
                m = binary_metrics(p == 1, g == 1)
            else:  # validation: chỉ tính Dice cho nhanh
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
    # DSC, Jaccard nhân 100 (%) để khớp cách trình bày trong bài báo; HD95, ASD theo pixel.
    summary = {}
    for k in keys:
        v = np.array([r[k] for r in rows]) * (100 if k in ("dsc", "jaccard") else 1)
        summary[k] = {"mean": float(v.mean()), "std": float(v.std())}
    (out_dir / "test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n===== KẾT QUẢ TẬP TEST =====")
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
    print(f"Mô hình {args.model}: {count_params(model):.2f}M tham số | thiết bị: {device}")

    if args.eval_only:
        ckpt = torch.load(args.ckpt or out_dir / "best.pth", map_location=device)
        model.load_state_dict(ckpt["model"])
        run_test(model, args, device, out_dir)
        return

    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    # ---- Dữ liệu ----
    split = Path(args.split_dir)
    train_names = read_list(split / ("train.txt" if args.labeled_ratio == 100
                                     else f"labeled_{args.labeled_ratio}.txt"))
    val_names = read_list(split / "val.txt")
    print(f"Ảnh huấn luyện có nhãn: {len(train_names)} | validation: {len(val_names)}")
    train_ds = BloodCellDataset(args.data_dir, train_names, train_transform(args.img_size))
    val_ds = BloodCellDataset(args.data_dir, val_names, eval_transform(args.img_size))
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=min(args.batch_size, len(train_ds)), shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              generator=g, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=args.num_workers)

    # ---- Tối ưu ----
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
        print(f"Tiếp tục từ vòng lặp {start_it} (best Dice = {best_dice:.4f})")

    log_f = open(out_dir / "log.csv", "a", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    if start_it == 0:
        log.writerow(["iter", "lr", "loss", "val_dice"])

    # ---- Vòng huấn luyện ----
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
        # Tính loss ở FP32 cho ổn định. L_sup = Cross-Entropy + Dice (giống DACL).
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

    # ---- Đánh giá tập test bằng checkpoint tốt nhất ----
    model.load_state_dict(torch.load(out_dir / "best.pth", map_location=device)["model"])
    run_test(model, args, device, out_dir)


if __name__ == "__main__":
    main()
