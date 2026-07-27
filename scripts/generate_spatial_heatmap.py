import os

# Thread safety controls to prevent server CPU lockups
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import time
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

data_file = PROCESSED_DIR / "epa_point_dataset_14months_20features.npz"
master_file = PROCESSED_DIR / "era5_california_1x1km_master.nc"
airnow_dir = BASE_DIR / "data" / "raw" / "epa" / "AirNow"
if not airnow_dir.exists():
  airnow_dir = BASE_DIR / "data" / "raw" / "AirNow"

print("🌍 STARTING STATE-WIDE SPATIAL POLLUTION HEAT MAP GENERATION 🌍")
print("=" * 75)

if not data_file.exists():
  raise FileNotFoundError(f"Could not locate 20-feature dataset at: {data_file}")

# ==============================================================================
# 2. LOAD DATA & RECREATE THE 0.852 R² DOMAIN SPLIT
# ==============================================================================
print("⏳ Loading 20-feature master dataset...")
data = np.load(data_file, allow_pickle=True)

X_full = np.vstack([data["X_train"], data["X_test"]])
y_full = np.concatenate([data["y_train"], data["y_test"]])
g_full = np.concatenate([data["groups_train"], data["groups_test"]])
feature_names = list(data["feature_names"])

print("🎯 Recreating 80/20 Literature Domain Split...")
X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X_full, y_full, g_full, test_size=0.20, random_state=42
)

# ==============================================================================
# 3. TRAIN THE OPTIMAL XGBOOST DIGITAL TWIN
# ==============================================================================
print("⏳ Training optimal 20-feature XGBoost model...")
start_train = time.time()

best_xgb = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=673,
    learning_rate=0.0327,
    max_depth=7,
    subsample=0.7100,
    colsample_bytree=0.6610,
    gamma=0.1324,
    reg_alpha=32.5183,
    reg_lambda=7.2963,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

best_xgb.fit(X_train, y_train)
print(f"✅ Model trained in {time.time() - start_train:.2f}s!")

print("⏳ Generating predictions across unseen test hours...")
y_pred = best_xgb.predict(X_test)
r2_overall = r2_score(y_test, y_pred)
print(f"🎯 Unseen Test Set R²: {r2_overall:.3f}")

# ==============================================================================
# 4. MAP GROUP IDS BACK TO PHYSICAL LATITUDE / LONGITUDE
# ==============================================================================
print("🔍 Extracting physical station coordinates from catalog metadata...")
airnow_files = sorted(list(airnow_dir.glob("*.nc")))

with nc.Dataset(airnow_files[0], "r") as a_ds:
  stn_lats_all = a_ds.variables["latitude"][:]
  stn_lons_all = a_ds.variables["longitude"][:]
  stn_ids_all = [str(sid).strip() for sid in a_ds.variables["site"][:]]

with nc.Dataset(master_file, "r") as m_ds:
  lat_key = "lat" if "lat" in m_ds.variables else "latitude"
  lon_key = "lon" if "lon" in m_ds.variables else "longitude"
  grid_lats = m_ds.variables[lat_key][:]
  grid_lons = m_ds.variables[lon_key][:]

# Build the exact valid index mask used during dataset creation
valid_mask = (
    (stn_lats_all >= np.min(grid_lats))
    & (stn_lats_all <= np.max(grid_lats))
    & (stn_lons_all >= np.min(grid_lons))
    & (stn_lons_all <= np.max(grid_lons))
)
valid_indices = np.where(valid_mask)[0]

stn_lats = stn_lats_all[valid_indices]
stn_lons = stn_lons_all[valid_indices]

# Map every test sample's group ID (gid) to its physical coordinates
test_lats = stn_lats[g_test]
test_lons = stn_lons[g_test]

# ==============================================================================
# 5. AGGREGATE MEAN DAYTIME POLLUTION BY STATION FOR MAPPING
# ==============================================================================
print("📊 Aggregating station-level climatological means for grid interpolation...")
df_map = pd.DataFrame({
    "gid": g_test,
    "lat": test_lats,
    "lon": test_lons,
    "True_NO2": y_test,
    "Pred_NO2": y_pred,
})

# Aggregate by station location to get spatial climatology across California
stn_summary = (
    df_map.groupby("gid")
    .agg({
        "lat": "first",
        "lon": "first",
        "True_NO2": "mean",
        "Pred_NO2": "mean",
        "gid": "count",
    })
    .rename(columns={"gid": "sample_count"})
    .reset_index()
)

print(
    f"📍 Interpolating across {len(stn_summary)} distinct California monitoring"
    " sites..."
)

# ==============================================================================
# 6. BUILD HIGH-RESOLUTION 2D SPATIAL INTERPOLATION GRID
# ==============================================================================
print("⚙️ Computing 300x300 cubic surface meshgrid over California...")
grid_lon_bins = np.linspace(
    stn_summary["lon"].min() - 0.2, stn_summary["lon"].max() + 0.2, 300
)
grid_lat_bins = np.linspace(
    stn_summary["lat"].min() - 0.2, stn_summary["lat"].max() + 0.2, 300
)
grid_x, grid_y = np.meshgrid(grid_lon_bins, grid_lat_bins)

points = stn_summary[["lon", "lat"]].values
true_vals = stn_summary["True_NO2"].values
pred_vals = stn_summary["Pred_NO2"].values

# Cubic interpolation for smooth atmospheric plumes, with nearest-neighbor edge filling
grid_true_cub = griddata(
    points, true_vals, (grid_x, grid_y), method="cubic"
)
grid_true_near = griddata(
    points, true_vals, (grid_x, grid_y), method="nearest"
)
grid_true = np.where(np.isnan(grid_true_cub), grid_true_near, grid_true_cub)

grid_pred_cub = griddata(
    points, pred_vals, (grid_x, grid_y), method="cubic"
)
grid_pred_near = griddata(
    points, pred_vals, (grid_x, grid_y), method="nearest"
)
grid_pred = np.where(np.isnan(grid_pred_cub), grid_pred_near, grid_pred_cub)

# ==============================================================================
# 7. RENDER THE 2-PANEL PUBLICATION HEAT MAP
# ==============================================================================
print("🎨 Rendering publication-grade colored spatial heat map...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), sharey=True)

# Define shared color limits so both maps use the exact same visual scale!
vmin, vmax = 2.0, np.percentile(true_vals, 98)
cmap = "turbo"  # Modern, publication-standard remote sensing colormap

# --- Left Panel: Observed Ground Telemetry ---
c1 = ax1.contourf(
    grid_x,
    grid_y,
    grid_true,
    levels=100,
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
    alpha=0.85,
)
sc1 = ax1.scatter(
    points[:, 0],
    points[:, 1],
    c=true_vals,
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
    edgecolor="black",
    linewidth=0.8,
    s=45,
    zorder=3,
    label="EPA Sensor Stations",
)
ax1.set_title(
    "Observed Ground Telemetry Surface\n(True Climatological Mean NO₂)",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
ax1.set_xlabel("Longitude (°W)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Latitude (°N)", fontsize=12, fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.4, color="white", zorder=1)
ax1.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

# --- Right Panel: 20-Feature XGBoost Digital Twin ---
c2 = ax2.contourf(
    grid_x,
    grid_y,
    grid_pred,
    levels=100,
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
    alpha=0.85,
)
sc2 = ax2.scatter(
    points[:, 0],
    points[:, 1],
    c=pred_vals,
    cmap=cmap,
    vmin=vmin,
    vmax=vmax,
    edgecolor="black",
    linewidth=0.8,
    s=45,
    zorder=3,
    label=f"XGBoost Twin (R²={r2_overall:.2f})",
)
ax2.set_title(
    "20-Feature XGBoost Digital Twin Surface\n(TEMPO + Weather + GIS Data"
    " Fusion)",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
ax2.set_xlabel("Longitude (°W)", fontsize=12, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.4, color="white", zorder=1)
ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

# --- Add City Callouts to highlight pollution basins! ---
cities = {
    "Los Angeles\nBasin": (-118.24, 34.05),
    "Fresno / Central\nValley": (-119.77, 36.75),
    "San Francisco\nBay": (-122.42, 37.77),
    "Sacramento": (-121.49, 38.58),
}

for ax in [ax1, ax2]:
  for city_name, (clon, clat) in cities.items():
    ax.annotate(
        city_name,
        xy=(clon, clat),
        xytext=(clon - 0.8, clat + 0.3),
        arrowprops=dict(
            facecolor="white", shrink=0.05, width=1.5, headwidth=6, edgecolor="black"
        ),
        fontsize=9.5,
        fontweight="bold",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9
        ),
        zorder=4,
    )

# --- Add Master Colorbar ---
cbar = fig.colorbar(c2, ax=[ax1, ax2], orientation="vertical", shrink=0.85, pad=0.02)
cbar.set_label(
    "Mean Ground-Level NO₂ Concentration (ppb)", fontsize=13, fontweight="bold"
)

fig.suptitle(
    "State-Wide High-Resolution Spatial Air Quality Downscaling\nNASA TEMPO"
    " Geostationary Data Fusion Architecture",
    fontsize=17,
    fontweight="bold",
    y=1.02,
)

out_file = OUTPUT_DIR / "california_spatial_pollution_map.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved State-Wide Spatial Heat Map to: {out_file}")
print("🎯 SPATIAL MAPPING FINISHED!")