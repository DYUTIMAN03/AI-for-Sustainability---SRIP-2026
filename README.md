# 🛰️ SRIP 2026 — Earth Observation Pipeline
**Land-use classification over the Delhi-NCR region** using Sentinel-2 satellite imagery and ESA WorldCover 2021 data.
This project implements a complete pipeline — from spatial filtering of raw satellite patches, through label construction using a global land-cover raster, to training and evaluating a CNN (ResNet18) for multi-class land-use classification.
---
## 📁 Project Structure
```
SRIP/
├── data/
│   └── Shapefile of the Delhi-NCR region (EPSG4326)/
│       ├── delhi_ncr_region.geojson      # Delhi-NCR boundary shapefile
│       ├── delhi_airshed.geojson         # Delhi Airshed boundary
│       ├── worldcover_bbox_delhi_ncr_2021.tif  # ESA WorldCover raster (~92 MB)
│       └── rgb/                          # 9,216 Sentinel-2 patches (128×128 px)
│
├── outputs/
│   ├── q1_grid_plot.png                  # NCR boundary + 60 km grid
│   ├── q1_filtered_points.png            # Filtered image centers map
│   ├── q1_filtered_images.csv            # 8,015 filtered images list
│   ├── q2_labelled_dataset.csv           # Full labelled dataset
│   ├── q2_train.csv / q2_test.csv        # 60/40 stratified splits
│   ├── q2_class_distribution.png         # Class distribution bar charts
│   ├── q3_confusion_matrix.png           # Confusion matrix
│   ├── q3_training_curves.png            # Loss & accuracy curves
│   └── q3_results.txt                    # Full metrics report
│
├── q1_spatial_filtering.py               # Q1 — Spatial reasoning & filtering
├── q2_label_construction.py              # Q2 — Labelling & dataset preparation
├── q3_cnn_training.py                    # Q3 — CNN training & evaluation
└── README.md
```
---
## ⚙️ Setup
### Prerequisites
- **Python 3.9+**
- NVIDIA GPU recommended (runs on CPU but slower)
### Install Dependencies
```bash
pip install geopandas rasterio matplotlib numpy pandas scipy torch torchvision scikit-learn Pillow seaborn
```
---
## 🚀 Usage
Run each script sequentially — each step depends on the previous one's outputs:
```bash
# Step 1: Filter satellite images to Delhi-NCR region
python q1_spatial_filtering.py
# Step 2: Construct land-cover labels from ESA WorldCover raster
python q2_label_construction.py
# Step 3: Train ResNet18 and evaluate
python q3_cnn_training.py
```
---
## 📝 Pipeline Details
### Q1: Spatial Reasoning & Data Filtering
- Loads the **Delhi-NCR shapefile** and reprojects to UTM (EPSG:32644) for metric gridding
- Creates a **60 × 60 km uniform grid** overlay (42 cells)
- Parses lat/lon from **9,216 image filenames** (`{lat}_{lon}.png`)
- Performs a **spatial join** to retain only images whose centers fall inside the NCR polygon
- **Result:** 9,216 → **8,015 images** (1,201 removed)
### Q2: Label Construction & Dataset Preparation
- For each filtered image, extracts a **128 × 128 pixel patch** from the WorldCover raster at the image's center coordinate
- Computes the **mode** (most frequent value) of each patch as the ESA class code
- Maps ESA codes to **4 simplified categories:**
| ESA Code(s) | Original Class | Simplified Label |
|-------------|---------------|-----------------|
| 10, 20, 30, 90, 95 | Tree, Shrub, Grass, Wetland, Mangroves | **Vegetation** |
| 40 | Cropland | **Cropland** |
| 50 | Built-up | **Built-up** |
| 80 | Water | **Water** |
| 60, 70, 100 | Bare, Snow, Moss | **Others** |
- Performs a **60/40 stratified train-test split** (Train: 4,809 / Test: 3,206)
### Q3: CNN Training & Evaluation
- **Model:** ResNet18 pretrained on ImageNet, fine-tuned (`layer4` + FC layer)
- **Augmentation:** Random flip, rotation (±15°), color jitter
- **Class imbalance handling:** Inverse-frequency weighted cross-entropy loss
- **Training:** 15 epochs, batch size 64, Adam (lr = 1e-4), StepLR scheduler
---
## 📊 Results
| Metric | Score |
|--------|-------|
| **Accuracy** | **92.3%** |
| F1-Score (macro) | 0.803 |
| F1-Score (weighted) | 0.925 |
### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Built-up | 0.89 | 0.91 | 0.90 | 711 |
| Cropland | 0.97 | 0.94 | 0.96 | 2,190 |
| Vegetation | 0.70 | 0.80 | 0.75 | 302 |
| Water | 0.43 | 1.00 | 0.60 | 3 |
### Key Observations
- **Cropland** achieves the highest F1 (0.96) — largest class with a distinct spectral signature
- **Built-up** performs well (F1 0.90) — clearly separable from agricultural land
- **Vegetation** shows moderate F1 (0.75) — some spectral overlap with Cropland
- **Water** has perfect recall but low precision — only 3 test samples; weighted loss prevents the model from ignoring it entirely
---
## 📂 Data Sources
| Dataset | Resolution | Source |
|---------|-----------|--------|
| Sentinel-2 RGB patches | 10 m/pixel, 128 × 128 px | Copernicus |
| ESA WorldCover 2021 | 10 m | [ESA WorldCover](https://esa-worldcover.org/en) |
| Delhi-NCR boundary | — | Shapefile (EPSG:4326) |
---
## 📜 License
This project was developed as part of the **SRIP 2026** selection task.
