import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# Try importing GeoPandas and Contextily for real-world road tile basemaps
try:
  import contextily as ctx
  import geopandas as gpd
  from shapely.geometry import Point

  HAS_GIS = True
except ImportError:
  HAS_GIS = False
  print(
      "⚠️ GeoPandas or Contextily not found. Will use standard Matplotlib grid"
      " fallback."
  )

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "epa_from_internet_daily"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

print(
    "🗺️ GENERATING GRIDDED CALIFORNIA MAP WITH HIGHWAYS & CLUSTERS (80/20"
    " SPLIT) 🗺️"
)
print("=" * 75)

# ==============================================================================
# 2. LOAD SITES & APPLY 5-CLUSTER K-MEANS SPLIT
# ==============================================================================
raw_file = RAW_DIR / "daily_42602_2024.csv"
if not raw_file.exists():
  raw_file = (
      BASE_DIR / "raw" / "epa_from_internet_daily" / "daily_42602_2024.csv"
  )

df_raw = pd.read_csv(
    raw_file,
    usecols=[
        "State Code",
        "County Code",
        "Site Num",
        "Latitude",
        "Longitude",
        "Local Site Name",
    ],
    low_memory=False,
)

# Explicit State Code 6 filter guarantees 100% California stations with zero Nevada leakage!
df_ca = df_raw[df_raw["State Code"] == 6].copy()
df_ca["Site_ID"] = (
    df_ca["State Code"].astype(str).str.zfill(2)
    + "-"
    + df_ca["County Code"].astype(str).str.zfill(3)
    + "-"
    + df_ca["Site Num"].astype(str).str.zfill(4)
)

sites_df = (
    df_ca.groupby("Site_ID")
    .agg({
        "Latitude": "first",
        "Longitude": "first",
        "Local Site Name": "first",
    })
    .reset_index()
)

# Fit 5 Spatial Clusters
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
sites_df["Cluster"] = kmeans.fit_predict(sites_df[["Latitude", "Longitude"]]) + 1

# Apply 80/20 Train/Test Split (Updated from 0.40 to 0.20!)
train_ids = []
for _, group in sites_df.groupby("Cluster"):
  tr, _ = train_test_split(group, test_size=0.20, random_state=42)
  train_ids.extend(tr["Site_ID"].tolist())

sites_df["Split"] = np.where(sites_df["Site_ID"].isin(train_ids), "Train", "Test")

print(f"✅ Processed {len(sites_df)} California monitoring sites across 5 clusters.")

# ==============================================================================
# 3. RENDER ADVANCED GIS MAP (WITH CARTO-DB ROADS & GRIDLINES)
# ==============================================================================
print("🎨 Rendering high-resolution map with underlying road network...")
fig, ax = plt.subplots(figsize=(13, 13))

cluster_colors = {
    1: "#1f77b4",  # Blue (LA Basin / Inland Empire)
    2: "#ff7f0e",  # Orange (NorCal / Bay Area)
    3: "#2ecc71",  # Green (Central Valley)
    4: "#d62728",  # Red (San Diego / Border)
    5: "#9467bd",  # Purple (Central Coast)
}

if HAS_GIS:
  # Convert DataFrame to GeoDataFrame with GPS coordinate system (EPSG:4326)
  geometry = [
      Point(xy) for xy in zip(sites_df["Longitude"], sites_df["Latitude"])
  ]
  gdf = gpd.GeoDataFrame(sites_df, crs="EPSG:4326", geometry=geometry)

  # Convert to Web Mercator (EPSG:3857) to perfectly align with OpenStreetMap/Carto tiles
  gdf_web = gdf.to_crs(epsg=3857)

  for cluster_num, group in gdf_web.groupby("Cluster"):
    # Plot Train (Circles)
    train_sub = group[group["Split"] == "Train"]
    train_sub.plot(
        ax=ax,
        color=cluster_colors[cluster_num],
        markersize=90,
        marker="o",
        edgecolor="black",
        linewidth=1.2,
        label=f"Cluster {cluster_num} (Train)",
        zorder=4,
    )

    # Plot Test (Triangles)
    test_sub = group[group["Split"] == "Test"]
    test_sub.plot(
        ax=ax,
        color=cluster_colors[cluster_num],
        markersize=110,
        marker="^",
        edgecolor="black",
        linewidth=1.2,
        label=f"Cluster {cluster_num} (Test)",
        zorder=5,
    )

  # Add CartoDB Positron Basemap (Exposes major highways, interstates, and city grids!)
  try:
    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.Positron,
        zoom=7,
        attribution="(c) OpenStreetMap contributors (c) CARTO",
    )
  except Exception as e:
    print(f"⚠️ Could not pull online map tiles ({e}). Using clean fallback.")

  ax.set_axis_on()

else:
  # Standard Matplotlib Fallback if GeoPandas isn't installed
  for cluster_num, group in sites_df.groupby("Cluster"):
    train_sub = group[group["Split"] == "Train"]
    ax.scatter(
        train_sub["Longitude"],
        train_sub["Latitude"],
        color=cluster_colors[cluster_num],
        s=80,
        marker="o",
        edgecolor="black",
        linewidth=1.0,
        label=f"Cluster {cluster_num} (Train)",
        zorder=4,
    )

    test_sub = group[group["Split"] == "Test"]
    ax.scatter(
        test_sub["Longitude"],
        test_sub["Latitude"],
        color=cluster_colors[cluster_num],
        s=100,
        marker="^",
        edgecolor="black",
        linewidth=1.0,
        label=f"Cluster {cluster_num} (Test)",
        zorder=5,
    )
  ax.set_xlim(-124.5, -113.5)
  ax.set_ylim(32.0, 42.5)

# Add prominent Lat/Long Grid lines across the map
ax.grid(
    True,
    which="major",
    color="#666666",
    linestyle="--",
    linewidth=0.8,
    alpha=0.6,
    zorder=2,
)

ax.set_title(
    "EPA Surface NO₂ Monitoring Sites - Spatially Aware K-Means"
    " Clusters\nCalifornia Gridded Road Network & Basin Coverage (80% Train /"
    " 20% Test)",
    fontsize=15,
    fontweight="bold",
    pad=18,
)
ax.set_xlabel("Longitude / Easting", fontsize=12, fontweight="bold")
ax.set_ylabel("Latitude / Northing", fontsize=12, fontweight="bold")

# Legend styling
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="black",
    framealpha=0.95,
    ncol=2,
    fontsize=10,
)

# Saved as a .jpg to prevent Overleaf cloud compilation timeouts!
out_file = OUTPUT_DIR / "california_gridded_cluster_map_80_20.jpg"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved Gridded Highway 80/20 Cluster Map to: {out_file}")
print("=" * 75)
print("🎯 MAP UPGRADE FINISHED!")