<div align="center">

# 🛰️ Delhi-NCR Land-Use Classification

### A Deep Learning Pipeline for Earth Observation

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-92.3%25-brightgreen?style=for-the-badge)
![F1 Score](https://img.shields.io/badge/F1_(weighted)-0.925-blue?style=for-the-badge)

*Classifying land-use patterns across the Delhi-NCR region using Sentinel-2 satellite imagery and ESA WorldCover 2021 — from raw patches to a trained CNN in three steps.*

---

[**Explore the Pipeline**](#-pipeline-overview) · [**View Results**](#-results) · [**Get Started**](#-quick-start)

</div>

---

## 🎯 What This Does

This project builds an **end-to-end land-use classification pipeline** that:

1. 🗺️ **Filters** 9,216 satellite patches down to those inside the Delhi-NCR boundary
2. 🏷️ **Labels** each patch using a global land-cover raster (ESA WorldCover 2021)
3. 🧠 **Trains** a fine-tuned ResNet18 to classify them into 4 land-use categories

> Built as part of the **SRIP 2026** selection task at IIT.

---

## 🔬 Pipeline Overview

```
┌─────────────────────┐     ┌─────────────────────────┐     ┌──────────────────────┐
│   Q1: SPATIAL        │     │   Q2: LABEL              │     │   Q3: MODEL           │
│   FILTERING          │────▶│   CONSTRUCTION           │────▶│   TRAINING            │
│                      │     │                          │     │                       │
│  • Load NCR shape    │     │  • Extract 128×128       │     │  • ResNet18 (pretrained│
│  • 60km grid overlay │     │    patches from raster   │     │    + fine-tuned)       │
│  • Spatial join      │     │  • Mode → ESA class      │     │  • Weighted CE loss    │
│  • 9216 → 8015 imgs  │     │  • Map to 4 categories   │     │  • 15 epochs training  │
│                      │     │  • 60/40 stratified split │     │  • Eval: Acc + F1      │
└─────────────────────┘     └─────────────────────────┘     └──────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- GPU recommended (runs on CPU, just slower)

### Install

```bash
pip install geopandas rasterio matplotlib numpy pandas scipy torch torchvision scikit-learn Pillow seaborn
```

### Run

```bash
python q1_spatial_filtering.py    # Step 1 — Filter images to NCR region
python q2_label_construction.py   # Step 2 — Build labels from WorldCover raster
python q3_cnn_training.py         # Step 3 — Train & evaluate ResNet18
```

> ⚠️ **Run sequentially** — each step depends on the previous step's outputs.

---

## 📂 Project Structure

```
SRIP/
│
├── 📜 q1_spatial_filtering.py        # Spatial reasoning & data filtering
├── 📜 q2_label_construction.py        # Label construction & dataset prep
├── 📜 q3_cnn_training.py              # CNN training & evaluation
│
├── 📁 data/
│   └── .../ 
│       ├── delhi_ncr_region.geojson   # NCR boundary (EPSG:4326)
│       ├── delhi_airshed.geojson      # Airshed boundary
│       ├── worldcover_*.tif           # ESA WorldCover raster (~92 MB)
│       └── rgb/                       # 9,216 Sentinel-2 patches (128×128)
│
└── 📁 outputs/                        # All generated outputs ↓
```

---

## 📊 Results

<div align="center">

### Overall Metrics

| Metric | Score |
|:------:|:-----:|
| 🎯 **Accuracy** | **92.3%** |
| 📐 F1 (macro) | 0.803 |
| ⚖️ F1 (weighted) | 0.925 |

</div>

### Per-Class Breakdown

| Class | Precision | Recall | F1-Score | Support | Verdict |
|:-----:|:---------:|:------:|:--------:|:-------:|:-------:|
| 🏙️ Built-up | 0.89 | 0.91 | **0.90** | 711 | ✅ Strong |
| 🌾 Cropland | 0.97 | 0.94 | **0.96** | 2,190 | ✅ Excellent |
| 🌿 Vegetation | 0.70 | 0.80 | **0.75** | 302 | ⚠️ Moderate |
| 💧 Water | 0.43 | 1.00 | **0.60** | 3 | ⚠️ Low support |

### 💡 Key Takeaways

- **Cropland** dominates with F1 = 0.96 — large sample size + distinct spectral signature
- **Built-up** areas are clearly separable from agricultural land (F1 = 0.90)
- **Vegetation** suffers from spectral overlap with Cropland (F1 = 0.75)
- **Water** has perfect recall but only 3 test samples — weighted loss keeps it alive

---

## 🧩 Methodology Deep Dive

<details>
<summary><b>Q1 — Spatial Filtering</b> (click to expand)</summary>

### How it works
1. Load `delhi_ncr_region.geojson` with GeoPandas
2. Reproject to **EPSG:32644** (UTM 44N) for metric-based gridding
3. Generate a **60 × 60 km uniform grid** → 42 cells
4. Parse `lat_lon.png` filenames to extract coordinates
5. Spatial join drops images outside the NCR polygon

### Numbers
| Before | After | Removed |
|:------:|:-----:|:-------:|
| 9,216 | **8,015** | 1,201 |

</details>

<details>
<summary><b>Q2 — Label Construction</b> (click to expand)</summary>

### ESA WorldCover Mapping

```
ESA Code    Original Class             →  Simplified
────────    ──────────────             ─  ──────────
10,20,30    Tree / Shrub / Grass       →  🌿 Vegetation
90,95       Wetland / Mangroves        →  🌿 Vegetation
40          Cropland                   →  🌾 Cropland
50          Built-up                   →  🏙️ Built-up
80          Water                      →  💧 Water
60,70,100   Bare / Snow / Moss         →  ⬜ Others
```

### Dataset Split
- **Train:** 4,809 images (60%)
- **Test:** 3,206 images (40%)
- Split is **stratified** to preserve class ratios

</details>

<details>
<summary><b>Q3 — Model Architecture</b> (click to expand)</summary>

### ResNet18 (Fine-tuned)

```
ImageNet Pretrained ResNet18
│
├── conv1 ─── layer1 ─── layer2 ─── layer3    ← FROZEN 🧊
│
├── layer4                                     ← TRAINABLE 🔥
│
└── FC: Dropout(0.3) → Linear(512, 4)         ← TRAINABLE 🔥
```

### Training Config
| Parameter | Value |
|-----------|-------|
| Epochs | 15 |
| Batch Size | 64 |
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Scheduler | StepLR (×0.5 every 5 epochs) |
| Loss | Weighted CrossEntropy |
| Augmentation | Flip, Rotation, Color Jitter |

</details>

---

## 🗂️ Output Files

| File | What's Inside |
|------|---------------|
| `q1_grid_plot.png` | NCR boundary with 60 km grid overlay |
| `q1_filtered_points.png` | Map: green (inside) vs red (outside) NCR |
| `q1_filtered_images.csv` | 8,015 filtered image filenames + coords |
| `q2_labelled_dataset.csv` | Complete labelled dataset |
| `q2_train.csv` / `q2_test.csv` | Stratified train/test splits |
| `q2_class_distribution.png` | Class distribution bar charts (3 panels) |
| `q3_confusion_matrix.png` | Raw + normalized confusion matrices |
| `q3_training_curves.png` | Loss & accuracy over 15 epochs |
| `q3_results.txt` | Full classification report + interpretation |

---

## 🌍 Data Sources

| Dataset | Resolution | Provider |
|---------|:----------:|----------|
| Sentinel-2 RGB patches | 10 m/px · 128×128 | [Copernicus](https://scihub.copernicus.eu/) |
| ESA WorldCover 2021 | 10 m | [ESA](https://esa-worldcover.org/) |
| Delhi-NCR Boundary | Vector | Shapefile (EPSG:4326) |

---

<div align="center">

**Built with** 🐍 Python · 🔥 PyTorch · 🌍 GeoPandas · 🛰️ Rasterio

*SRIP 2026 Selection Task*

</div>
