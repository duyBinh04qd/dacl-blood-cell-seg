# DACL for Blood Cell Segmentation in Microscopy Images

Research project: **Applying "Dual Agreement Consistency Learning" (DACL) to semi-supervised blood cell segmentation in microscopy images.**

DACL ([Wang et al., 2026](https://arxiv.org/abs/2606.25254)) is a semi-supervised framework that jointly trains UNeXt (lightweight CNN, ~1.5M parameters) and Swin-Unet (Transformer), combining:

- **CPS** – cross pseudo supervision between the two networks,
- **ICT** – interpolation (mixup) consistency with an EMA teacher,
- **DAC** – pixel-wise dual-agreement consistency loss (KL + entropy).

Only UNeXt is used at inference time.

## Dataset

**Blood Cell Segmentation** dataset (1,328 images + 1,328 masks) by Depto et al., *Tissue and Cell* 73 (2021) 101653 –
download from <https://github.com/Deponker/Blood-cell-segmentation>.

The data is **not** included in this repository.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Usage

### Kaggle (self-contained notebooks, no cloning needed)

| Notebook | Model |
|---|---|
| [notebooks/fs-unext.ipynb](notebooks/fs-unext.ipynb) | Fully supervised UNeXt (5/10/20/100% labels) |
| [notebooks/fs-swinunet.ipynb](notebooks/fs-swinunet.ipynb) | Fully supervised Swin-Unet (5/10/20/100% labels) |
| [notebooks/ssl-mt.ipynb](notebooks/ssl-mt.ipynb) | Mean Teacher: UNeXt student + EMA teacher (5/10% labels) |
| [notebooks/ssl-dct.ipynb](notebooks/ssl-dct.ipynb) | Deep Co-Training: two UNeXt networks (5/10% labels) |
| [notebooks/ssl-cps.ipynb](notebooks/ssl-cps.ipynb) | Cross Pseudo Supervision: two UNeXt networks (5/10% labels) |
| [notebooks/ssl-ict.ipynb](notebooks/ssl-ict.ipynb) | Interpolation Consistency Training: UNeXt student + EMA teacher (5/10% labels) |

Each notebook covers: configuration → data preparation → **data augmentation** (configurable, with previews) →
model → training → results table, training curves and prediction overlays.
Only `CFG.RAW_DIR` needs to point to your Kaggle Dataset (`<dataset>/Original` + `<dataset>/Mask`).
All notebooks share the same split (`SPLIT_SEED = 2026`), so the labeled / unlabeled / test sets are identical
across methods. Semi-supervised notebooks use 1 labeled + 4 unlabeled images per iteration, as in the DACL paper.

### Local scripts

```bash
# 1. Normalize the raw data (pair *_image / *_mask, resize to 640x480, binarize masks)
python scripts/prepare_dataset.py --raw <raw-data-folder> --out data/processed
# 2. 70/10/20 split + 2/5/10/20% labeled lists
python scripts/split_dataset.py --images data/processed/images --out data/splits
# 3. Fully supervised training
python train_supervised.py --model unext     --labeled_ratio 100
python train_supervised.py --model swin_unet --labeled_ratio 10 --pretrained imagenet
# 4. Re-evaluate on the test set only
python train_supervised.py --model unext --labeled_ratio 100 --eval_only
```

Each run writes to `runs/<model>_<ratio>p_s<seed>/`: `best.pth`, `last.pth`,
`log.csv`, `config.json`, `test_per_image.csv`, `test_summary.json`.

## Project structure

```
├── docs/latex/               # project proposal / report (LaTeX, Vietnamese)
├── notebooks/                # Kaggle notebooks
├── scripts/                  # data preparation, inspection & splitting
├── src/
│   ├── data/                 # Dataset, augmentation
│   ├── models/               # UNeXt, Swin-Unet
│   ├── dacl/losses.py        # L_sup, L_cps, L_ict, L_dac, EMA
│   └── utils/                # metrics, helpers
├── train_supervised.py       # fully supervised training
├── requirements.txt
└── README.md
```

## Citation

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
