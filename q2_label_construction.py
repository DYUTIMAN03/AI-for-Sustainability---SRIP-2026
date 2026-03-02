"""
Q2: Label Construction & Dataset Preparation [6 Marks]
- Extract 128x128 land-cover patches from land_cover.tif
- Assign labels using dominant (mode) land-cover class
- Map ESA class codes to simplified categories
- 60/40 train-test split with class distribution visualization
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "Shapefile of the Delhi-NCR region (EPSG4326)")
TIFF_PATH = os.path.join(DATA_DIR, "worldcover_bbox_delhi_ncr_2021.tif")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FILTERED_CSV = os.path.join(OUTPUT_DIR, "q1_filtered_images.csv")

# ── ESA WorldCover class mapping ───────────────────────────────────────────────
ESA_TO_SIMPLE = {
    10: "Vegetation",    # Tree cover
    20: "Vegetation",    # Shrubland
    30: "Vegetation",    # Grassland
    40: "Cropland",      # Cropland
    50: "Built-up",      # Built-up
    60: "Others",        # Bare / sparse vegetation
    70: "Others",        # Snow and Ice
    80: "Water",         # Permanent water bodies
    90: "Vegetation",    # Herbaceous wetland
    95: "Vegetation",    # Mangroves
    100: "Others",       # Moss and lichen
}

ESA_CLASS_NAMES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland",
    40: "Cropland", 50: "Built-up", 60: "Bare/sparse",
    70: "Snow/Ice", 80: "Water", 90: "Herbaceous wetland",
    95: "Mangroves", 100: "Moss/lichen",
}

# ── 1. Load filtered images list ───────────────────────────────────────────────
print("Loading filtered image list from Q1...")
df = pd.read_csv(FILTERED_CSV)
print(f"  Filtered images: {len(df)}")

# ── 2. Open land_cover.tif and extract patches ────────────────────────────────
print("\nOpening land_cover.tif raster...")
src = rasterio.open(TIFF_PATH)
print(f"  Raster CRS: {src.crs}")
print(f"  Raster size: {src.width} x {src.height}")
print(f"  Raster resolution: {src.res}")
print(f"  Raster bounds: {src.bounds}")

PATCH_SIZE = 128  # pixels, matching the satellite image size

successful_indices = []
labels_list = []
esa_codes_list = []
skipped = 0

print(f"\nExtracting {PATCH_SIZE}x{PATCH_SIZE} patches for {len(df)} images...")
for idx, row in df.iterrows():
    lat, lon = row["lat"], row["lon"]

    # rasterio.index(x, y) where x=lon, y=lat for EPSG:4326
    try:
        row_px, col_px = src.index(lon, lat)
    except Exception:
        skipped += 1
        continue

    # Calculate window centered on the point
    half = PATCH_SIZE // 2
    win_col = col_px - half
    win_row = row_px - half

    # Bounds check
    if win_col < 0 or win_row < 0 or win_col + PATCH_SIZE > src.width or win_row + PATCH_SIZE > src.height:
        skipped += 1
        continue

    # Read the patch
    window = Window(win_col, win_row, PATCH_SIZE, PATCH_SIZE)
    patch = src.read(1, window=window)

    # Compute mode (most frequent land cover class)
    values = patch.flatten()
    values = values[values > 0]  # Exclude nodata (0)
    if len(values) == 0:
        skipped += 1
        continue

    mode_result = stats.mode(values, keepdims=True)
    mode_class = int(mode_result.mode[0])

    # Map to simplified category
    simple_label = ESA_TO_SIMPLE.get(mode_class, "Others")

    successful_indices.append(idx)
    labels_list.append(simple_label)
    esa_codes_list.append(mode_class)

    if len(successful_indices) % 2000 == 0:
        print(f"    Labelled {len(successful_indices)} images...")

src.close()

print(f"\n  Successfully labelled: {len(labels_list)}")
print(f"  Skipped (out of bounds/nodata): {skipped}")

# ── 3. Build labelled dataset ──────────────────────────────────────────────────
df_labelled = df.loc[successful_indices].copy()
df_labelled["esa_class"] = esa_codes_list
df_labelled["label"] = labels_list
df_labelled = df_labelled.reset_index(drop=True)

print(f"\n  Final labelled dataset: {len(df_labelled)} images")
print(f"\n  ESA class distribution:")
for code, count in sorted(Counter(esa_codes_list).items()):
    name = ESA_CLASS_NAMES.get(code, f"Unknown({code})")
    print(f"    {code:>3} ({name:>20}): {count}")

print(f"\n  Simplified label distribution:")
for label, count in sorted(Counter(labels_list).items()):
    print(f"    {label:>12}: {count}")

# ── 4. 60/40 stratified train-test split ──────────────────────────────────────
print("\nPerforming 60/40 stratified train-test split...")
train_df, test_df = train_test_split(
    df_labelled, test_size=0.4, random_state=42, stratify=df_labelled["label"]
)
print(f"  Train set: {len(train_df)}")
print(f"  Test set:  {len(test_df)}")

# Save
train_df.to_csv(os.path.join(OUTPUT_DIR, "q2_train.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "q2_test.csv"), index=False)
df_labelled.to_csv(os.path.join(OUTPUT_DIR, "q2_labelled_dataset.csv"), index=False)
print("  Saved train/test CSVs to outputs/")

# ── 5. Visualize class distribution ───────────────────────────────────────────
print("\nGenerating class distribution visualization...")

categories = sorted(df_labelled["label"].unique())
train_counts = train_df["label"].value_counts().reindex(categories, fill_value=0)
test_counts = test_df["label"].value_counts().reindex(categories, fill_value=0)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Color map
colors = {"Built-up": "#e74c3c", "Cropland": "#f39c12", "Vegetation": "#27ae60", "Water": "#3498db", "Others": "#95a5a6"}
bar_colors = [colors.get(c, "#333333") for c in categories]

# Overall distribution
overall_counts = df_labelled["label"].value_counts().reindex(categories, fill_value=0)
axes[0].bar(categories, overall_counts, color=bar_colors, edgecolor="black", linewidth=0.5)
axes[0].set_title("Overall Class Distribution", fontsize=13, fontweight="bold")
axes[0].set_ylabel("Count", fontsize=11)
for i, v in enumerate(overall_counts):
    axes[0].text(i, v + 20, str(v), ha="center", fontsize=9, fontweight="bold")

# Train distribution
axes[1].bar(categories, train_counts, color=bar_colors, edgecolor="black", linewidth=0.5)
axes[1].set_title(f"Train Set ({len(train_df)} images, 60%)", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Count", fontsize=11)
for i, v in enumerate(train_counts):
    axes[1].text(i, v + 15, str(v), ha="center", fontsize=9, fontweight="bold")

# Test distribution
axes[2].bar(categories, test_counts, color=bar_colors, edgecolor="black", linewidth=0.5)
axes[2].set_title(f"Test Set ({len(test_df)} images, 40%)", fontsize=13, fontweight="bold")
axes[2].set_ylabel("Count", fontsize=11)
for i, v in enumerate(test_counts):
    axes[2].text(i, v + 10, str(v), ha="center", fontsize=9, fontweight="bold")

for ax in axes:
    ax.tick_params(axis="x", rotation=30)
    ax.set_xlabel("Land-Use Category", fontsize=11)

plt.suptitle("Land-Use Class Distribution (ESA WorldCover 2021 -> Simplified)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "q2_class_distribution.png"), dpi=150, bbox_inches="tight")
print(f"  Class distribution plot saved to outputs/q2_class_distribution.png")

plt.close("all")
print("\n[DONE] Q2 complete!")
