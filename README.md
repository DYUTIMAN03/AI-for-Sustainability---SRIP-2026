# 🛰️ Delhi-NCR Land-Use Classification
> A Deep Learning Pipeline for Earth Observation — **SRIP 2026 Selection Task**
> 
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-92.3%25-brightgreen?style=flat-square)
![F1](https://img.shields.io/badge/F1_(weighted)-0.925-blue?style=flat-square)

Classifying land-use patterns across the **Delhi-NCR region** using Sentinel-2 satellite imagery and ESA WorldCover 2021 — from raw patches to a trained CNN in three steps.
---
## 🎯 What This Does
| Step | Script | Description |
|:----:|--------|-------------|
| **Q1** | `q1_spatial_filtering.py` | 🗺️ Filters 9,216 satellite patches to those inside Delhi-NCR → **8,015 images** |
| **Q2** | `q2_label_construction.py` | 🏷️ Labels each patch from the ESA WorldCover raster → **4 land-use classes** |
| **Q3** | `q3_cnn_training.py` | 🧠 Trains a fine-tuned ResNet18 → **92.3% accuracy** |
---
## 🚀 Quick Start
**Prerequisites:** Python 3.9+ · GPU recommended but not required
```bash
# Install dependencies
pip install geopandas rasterio matplotlib numpy pandas scipy torch torchvision scikit-learn Pillow seaborn
# Run the pipeline (must be sequential)
python q1_spatial_filtering.py
python q2_label_construction.py
python q3_cnn_training.py
```
---
## 📂 Project Structure
```
SRIP/
├── q1_spatial_filtering.py           # Spatial reasoning & data filtering
├── q2_label_construction.py          # Label construction & dataset prep
├── q3_cnn_training.py                # CNN training & evaluation
├── README.md
│
├── data/
│   ├── delhi_ncr_region.geojson      # NCR boundary (EPSG:4326)
│   ├── delhi_airshed.geojson         # Airshed boundary
│   ├── worldcover_*.tif              # ESA WorldCover raster (~92 MB)
│   └── rgb/                          # 9,216 Sentinel-2 patches (128×128 px)
│
└── outputs/                          # All generated outputs (see below)
```
---
## 🔬 Pipeline Details
### Q1 — Spatial Filtering
- Loads `delhi_ncr_region.geojson` and reprojects to **EPSG:32644** (UTM 44N) for metric gridding
- Creates a **60 × 60 km uniform grid** overlay (42 cells)
- Parses lat/lon from each image filename (`{lat}_{lon}.png`)
- Spatial join keeps only points inside the NCR polygon

| Metric | Value |
|--------|-------|
| Images before filtering | 9,216 |
| Images after filtering | **8,015** |
| Images removed | 1,201 |
| Grid size | 60 × 60 km |
| Grid cells | 42 |
### Q2 — Label Construction
For each filtered image, extracts a 128×128 patch from the WorldCover raster, computes the **mode** (most frequent class), and maps it:
| ESA Code(s)       | Original Class                              | → Simplified    | Count |
|-------------------|---------------------------------------------|-----------------|-------|
| 10, 20, 30, 90, 95| Tree / Shrub / Grass / Wetland / Mangroves  | 🌿 **Vegetation** | 755   |
| 40                | Cropland                                    | 🌾 **Cropland** | 5,474 |
| 50                | Built-up                                    | 🏙️ **Built-up** | 1,779 |
| 80                | Water                                       | 💧 **Water**    | 7     |
| 60, 70, 100       | Bare / Snow / Moss                          | ⬜ **Others**   | —     |

**Stratified 60/40 split** → Train: 4,809 · Test: 3,206
### Q3 — Model Training
**Architecture:** ResNet18 (ImageNet pretrained)
- Layers `conv1` → `layer3`: **frozen** 🧊
- `layer4` + FC (`Dropout(0.3) → Linear(512, 4)`): **trainable** 🔥
**Config:** 15 epochs · batch 64 · Adam (lr=1e-4) · StepLR (×0.5 every 5 epochs) · weighted CrossEntropy
---
## 📊 Results
| Metric | Score |
|:------:|:-----:|
| 🎯 **Accuracy** | **92.3%** |
| 📐 F1 (macro) | 0.803 |
| ⚖️ F1 (weighted) | 0.925 |
### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support | |
|:-----:|:---------:|:------:|:--------:|:-------:|:-:|
| 🏙️ Built-up | 0.89 | 0.91 | **0.90** | 711 | ✅ |
| 🌾 Cropland | 0.97 | 0.94 | **0.96** | 2,190 | ✅ |
| 🌿 Vegetation | 0.70 | 0.80 | **0.75** | 302 | ⚠️ |
| 💧 Water | 0.43 | 1.00 | **0.60** | 3 | ⚠️ |
### 💡 Key Takeaways
- **Cropland** → F1 = 0.96 — largest class with distinct spectral signature
- **Built-up** → F1 = 0.90 — clearly separable from agricultural land
- **Vegetation** → F1 = 0.75 — spectral overlap with Cropland causes some confusion
- **Water** → Perfect recall, but only 3 test samples — weighted loss prevents it from being ignored
---
## 🗂️ Output Files
| File | Description |
|------|-------------|
| `q1_grid_plot.png` | NCR boundary with 60 km grid overlay |
| `q1_filtered_points.png` | Image centers: green (inside) vs red (outside) |
| `q1_filtered_images.csv` | 8,015 filtered images with coordinates |
| `q2_labelled_dataset.csv` | Complete labelled dataset |
| `q2_train.csv` / `q2_test.csv` | Stratified train/test splits |
| `q2_class_distribution.png` | Class distribution bar charts |
| `q3_confusion_matrix.png` | Raw + normalized confusion matrices |
| `q3_training_curves.png` | Loss & accuracy over 15 epochs |
| `q3_results.txt` | Full classification report + interpretation |
---
## 🌍 Data Sources
| Dataset | Resolution | Source |
|---------|:----------:|--------|
| Sentinel-2 RGB patches | 10 m/px · 128×128 | [Copernicus](https://scihub.copernicus.eu/) |
| ESA WorldCover 2021 | 10 m | [ESA WorldCover](https://esa-worldcover.org/) |
| Delhi-NCR Boundary | Vector | Shapefile (EPSG:4326) |
---
*Built with 🐍 Python · 🔥 PyTorch · 🌍 GeoPandas · 🛰️ Rasterio*
