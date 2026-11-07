# DACL cho phân đoạn tế bào máu trong ảnh hiển vi

Đồ án khoa học: **Nghiên cứu và ứng dụng mô hình "Dual Agreement Consistency Learning" (DACL) cho bài toán phân đoạn tế bào máu trong ảnh hiển vi.**

DACL ([Wang et al., 2026](https://arxiv.org/abs/2606.25254)) là khung học bán giám sát huấn luyện đồng thời UNeXt (CNN nhẹ, ~1.5M tham số) và Swin-Unet (Transformer), kết hợp:

- **CPS** – giám sát giả chéo giữa hai mạng,
- **ICT** – nhất quán nội suy (mixup) với giáo viên EMA,
- **DAC** – mất mát đồng thuận kép (KL + entropy) theo từng điểm ảnh.

Khi suy luận chỉ dùng UNeXt.

## Dữ liệu

Bộ dữ liệu **Blood Cell Segmentation** (1 328 ảnh + 1 328 mask) của Depto et al., *Tissue and Cell* 73 (2021) 101653 – tải tại
<https://github.com/Deponker/Blood-cell-segmentation>.

Dữ liệu **không** được đưa lên repo này. Sau khi tải, sắp xếp như sau:

```
data/raw/images/   # ảnh gốc
data/raw/masks/    # mask (cùng tên tệp với ảnh)
```

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Sử dụng

```bash
# 1. Chuẩn hóa dữ liệu gốc (ghép cặp *_image / *_mask, thu nhỏ 640x480, nhị phân hóa mask)
python scripts/prepare_dataset.py --raw <thu-muc-du-lieu-goc> --out data/processed
# 2. Chia 70/10/20 + danh sách ảnh có nhãn 5/10/20%
python scripts/split_dataset.py --images data/processed/images --out data/splits
# 3. Huấn luyện có giám sát (FS)
python train_supervised.py --model unext     --labeled_ratio 100
python train_supervised.py --model swin_unet --labeled_ratio 10 --pretrained imagenet
# 4. Chỉ đánh giá lại trên tập test
python train_supervised.py --model unext --labeled_ratio 100 --eval_only
```

### Chạy trên Kaggle (notebook độc lập, không cần clone code)

| Notebook | Mô hình |
|---|---|
| [notebooks/FS_UNeXt.ipynb](notebooks/FS_UNeXt.ipynb) | UNeXt huấn luyện có giám sát |
| [notebooks/FS_SwinUnet.ipynb](notebooks/FS_SwinUnet.ipynb) | Swin-Unet huấn luyện có giám sát |

Mỗi notebook gồm: cấu hình → chuẩn bị dữ liệu → **data augmentation** (cấu hình + xem trực quan) →
mô hình → huấn luyện ở các mức nhãn 5/10/20/100% → bảng kết quả, đường cong, ảnh dự đoán.
Chỉ cần sửa `CFG.RAW_DIR` cho đúng đường dẫn Kaggle Dataset.

Kết quả mỗi lần chạy nằm trong `runs/<model>_<ratio>p_s<seed>/`: `best.pth`, `last.pth`,
`log.csv`, `config.json`, `test_per_image.csv`, `test_summary.json`.

## Cấu trúc thư mục

```
├── docs/latex/               # đề cương / báo cáo LaTeX
├── notebooks/                # notebook chạy trên Kaggle
├── scripts/                  # chuẩn hóa, kiểm tra & chia dữ liệu
├── src/
│   ├── data/                 # Dataset, tăng cường dữ liệu
│   ├── models/               # UNeXt, Swin-Unet
│   ├── dacl/losses.py        # L_sup, L_cps, L_ict, L_dac, EMA
│   └── utils/                # chỉ số đánh giá, tiện ích
├── train_supervised.py       # huấn luyện có giám sát
├── requirements.txt
└── README.md
```

## Trích dẫn

```bibtex
@article{wang2026dacl,
  title   = {Dual Agreement Consistency Learning for Semi-Supervised Fetal Ultrasound Segmentation},
  author  = {Wang, Fangyijie and Silvestre, Gu{\'e}nol{\'e} and Wang, Ziyang and Curran, Kathleen M.},
  journal = {arXiv preprint arXiv:2606.25254},
  year    = {2026}
}
@article{depto2021blood,
  title   = {Automatic segmentation of blood cells from microscopic slides: A comparative analysis},
  author  = {Depto, Deponker Sarker and others},
  journal = {Tissue and Cell}, volume = {73}, pages = {101653}, year = {2021}
}
```
