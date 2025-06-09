# 🖼️🔍CNN & BatchNorm on CIFAR-10 - Course Project II

This project explores convolutional neural networks (CNNs) on the CIFAR-10 dataset with a focus on the effects of **Batch Normalization** and **Dropout**. It is designed for an academic coursework submission.

## 📁 Project Structure

```
.
├── data/               # Data loading and transforms (loaders.py)
├── models/             # Model architectures (VGG_A, VGG_BatchNorm, etc.)
├── scripts/            # Training and analysis scripts
├── utils/              # Utility functions (loss tracker, init, etc.)
├── outputs/            # Generated plots (loss curves, landscape analysis)
├── checkpoints/        # Saved model weights
├── requirements.txt    # Python dependencies
├── CNN_Project_Report.docx   # Final report
└── README.md           # This file
```

---

## 🔧 Setup (Windows / Linux / macOS)

### 1. Create environment (optional but recommended)
```bash
python -m venv venv
venv\Scripts\activate       # Windows
# OR
source venv/bin/activate    # Linux/macOS
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Training Scripts

Train various CNN models and compare their performance:

```bash
python scripts/train_basic.py             # Baseline CNN (BasicCNN)
python scripts/train_bn_improved.py       # CNN with BatchNorm + Dropout
python scripts/train_vgg_a.py             # VGG_A (no BatchNorm)
python scripts/train_vgg_bn.py            # VGG with BatchNorm
```

All scripts will:
- Print per-epoch training loss & test accuracy
- Save model to `checkpoints/`
- Save loss & accuracy curves to `outputs/`

---

## 📊 Loss Landscape Analysis (2.3 Bonus Part)

This evaluates how BatchNorm affects optimization stability by comparing loss curves across different learning rates.

```bash
python scripts/landscape_vgg_a.py   # Trains both VGG_A and VGG_BatchNorm
```

This script automatically:
- Trains both models with learning rates: `[1e-4, 5e-4, 1e-3, 2e-3]`
- Tracks per-epoch training loss
- Computes max_curve and min_curve across learning rates
- Saves landscape plots to `outputs/`:
  - `vgg_a_landscape_true.png`
  - `vgg_batchnorm_landscape_true.png`

---
