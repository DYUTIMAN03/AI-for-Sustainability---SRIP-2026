"""
Q1: Spatial Reasoning & Data Filtering [4 Marks]
- Plot the Delhi-NCR shapefile using matplotlib and overlay a 60×60 km uniform grid
- Filter satellite images whose center coordinates fall inside the region
- Report the total number of images before and after filtering
"""

import os
import re
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, Point
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "Shapefile of the Delhi-NCR region (EPSG4326)")
GEOJSON_PATH = os.path.join(DATA_DIR, "delhi_ncr_region.geojson")
RGB_DIR = os.path.join(DATA_DIR, "rgb")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Load Delhi-NCR shapefile ────────────────────────────────────────────────
print("Loading Delhi-NCR shapefile...")
ncr = gpd.read_file(GEOJSON_PATH)
print(f"  CRS: {ncr.crs}")
print(f"  Shape: {ncr.shape}")

# ── 2. Reproject to EPSG:32644 (UTM 44N) for metric grid ──────────────────────
ncr_utm = ncr.to_crs(epsg=32644)
bounds = ncr_utm.total_bounds  # minx, miny, maxx, maxy
print(f"  UTM bounds: {bounds}")

# ── 3. Create 60×60 km uniform grid ───────────────────────────────────────────
GRID_SIZE = 60_000  # 60 km in metres

# Align grid to nice boundaries
xmin = np.floor(bounds[0] / GRID_SIZE) * GRID_SIZE
ymin = np.floor(bounds[1] / GRID_SIZE) * GRID_SIZE
xmax = np.ceil(bounds[2] / GRID_SIZE) * GRID_SIZE
ymax = np.ceil(bounds[3] / GRID_SIZE) * GRID_SIZE

grid_cells = []
x = xmin
while x < xmax:
    y = ymin
    while y < ymax:
        cell = box(x, y, x + GRID_SIZE, y + GRID_SIZE)
        grid_cells.append(cell)
        y += GRID_SIZE
    x += GRID_SIZE

grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:32644")
print(f"  Grid cells created: {len(grid_cells)}")

# ── 4. Plot shapefile with grid overlay ────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(12, 12))

# Plot NCR boundary
ncr_utm.boundary.plot(ax=ax, color="black", linewidth=1.5, label="Delhi-NCR Boundary")

# Plot grid
grid_gdf.boundary.plot(ax=ax, color="blue", linewidth=0.5, alpha=0.6, linestyle="--")

ax.set_title("Delhi-NCR Region with 60×60 km Uniform Grid", fontsize=14, fontweight="bold")
ax.set_xlabel("Easting (m)", fontsize=11)
ax.set_ylabel("Northing (m)", fontsize=11)

# Legend
legend_handles = [
    mpatches.Patch(edgecolor="black", facecolor="none", linewidth=1.5, label="Delhi-NCR Boundary"),
    mpatches.Patch(edgecolor="blue", facecolor="none", linewidth=0.5, linestyle="--", label="60×60 km Grid"),
]
ax.legend(handles=legend_handles, loc="upper right", fontsize=10)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "q1_grid_plot.png"), dpi=150, bbox_inches="tight")
print(f"  Grid plot saved to outputs/q1_grid_plot.png")

# ── 5. Parse image coordinates ─────────────────────────────────────────────────
print("\nParsing image filenames...")
all_files = [f for f in os.listdir(RGB_DIR) if f.endswith(".png")]
total_before = len(all_files)
print(f"  Total images before filtering: {total_before}")

coords = []
for f in all_files:
    name = f.replace(".png", "")
    parts = name.split("_")
    if len(parts) == 2:
        try:
            lat, lon = float(parts[0]), float(parts[1])
            coords.append({"filename": f, "lat": lat, "lon": lon})
        except ValueError:
            pass

print(f"  Successfully parsed coordinates: {len(coords)}")

# ── 6. Filter images inside Delhi-NCR ──────────────────────────────────────────
points_gdf = gpd.GeoDataFrame(
    coords,
    geometry=[Point(c["lon"], c["lat"]) for c in coords],
    crs="EPSG:4326",
)

# Spatial join: keep only points inside the NCR polygon
ncr_4326 = ncr.copy()
filtered = gpd.sjoin(points_gdf, ncr_4326, predicate="within")
total_after = len(filtered)

print(f"\n{'='*50}")
print(f"  Total images BEFORE filtering: {total_before}")
print(f"  Total images AFTER  filtering: {total_after}")
print(f"  Images removed: {total_before - total_after}")
print(f"{'='*50}")

# ── 7. Save filtered file list ─────────────────────────────────────────────────
filtered_files = filtered["filename"].tolist()
filtered_coords = filtered[["filename", "lat", "lon"]].reset_index(drop=True)
filtered_coords.to_csv(os.path.join(OUTPUT_DIR, "q1_filtered_images.csv"), index=False)
print(f"\n  Filtered image list saved to outputs/q1_filtered_images.csv")

# ── 8. Plot filtered points on the map ─────────────────────────────────────────
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 12))
ncr.boundary.plot(ax=ax2, color="black", linewidth=1.5)

# Plot all points
points_gdf.plot(ax=ax2, color="red", markersize=0.5, alpha=0.3, label="Outside NCR")
# Plot filtered points
filtered_plot = gpd.GeoDataFrame(
    filtered[["filename", "lat", "lon"]],
    geometry=[Point(row["lon"], row["lat"]) for _, row in filtered.iterrows()],
    crs="EPSG:4326",
)
filtered_plot.plot(ax=ax2, color="green", markersize=0.5, alpha=0.5, label="Inside NCR")

ax2.set_title("Satellite Image Centers: Filtered by Delhi-NCR", fontsize=14, fontweight="bold")
ax2.set_xlabel("Longitude", fontsize=11)
ax2.set_ylabel("Latitude", fontsize=11)
legend_handles2 = [
    mpatches.Patch(color="green", label=f"Inside NCR ({total_after})"),
    mpatches.Patch(color="red", label=f"Outside NCR ({total_before - total_after})"),
    mpatches.Patch(edgecolor="black", facecolor="none", linewidth=1.5, label="NCR Boundary"),
]
ax2.legend(handles=legend_handles2, loc="upper right", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "q1_filtered_points.png"), dpi=150, bbox_inches="tight")
print(f"  Filtered points plot saved to outputs/q1_filtered_points.png")

plt.close("all")
print("\n[DONE] Q1 complete!")
