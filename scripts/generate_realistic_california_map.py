import os

# Thread safety controls to prevent server CPU lockups
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
from pathlib import Path
import time
import urllib.request
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
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

print("🌍 STARTING REALISTIC CALIFORNIA ATMOSPHERIC DISPERSION MAPPING 🌍")
print("=" * 75)

if not data_file.exists():
  raise FileNotFoundError(f"Could not locate 20-feature dataset at: {data_file}")

# ==============================================================================
# 2. LOAD DATA & TRAIN THE OPTIMAL XGBOOST DIGITAL TWIN
# ==============================================================================
print("⏳ Loading 20-feature master dataset...")
data = np.load(data_file, allow_pickle=True)

X_full = np.vstack([data["X_train"], data["X_test"]])
y_full = np.concatenate([data["y_train"], data["y_test"]])
g_full = np.concatenate([data["groups_train"], data["groups_test"]])

print("🎯 Recreating 80/20 Literature Domain Split...")
X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X_full, y_full, g_full, test_size=0.20, random_state=42
)

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

y_pred = best_xgb.predict(X_test)
r2_overall = r2_score(y_test, y_pred)
print(f"🎯 Unseen Test Set R²: {r2_overall:.3f}")

# ==============================================================================
# 3. EXTRACT PHYSICAL STATION COORDINATES
# ==============================================================================
print("🔍 Extracting physical station coordinates from AirNow catalog...")
airnow_files = sorted(list(airnow_dir.glob("*.nc")))

with nc.Dataset(airnow_files[0], "r") as a_ds:
  stn_lats_all = a_ds.variables["latitude"][:]
  stn_lons_all = a_ds.variables["longitude"][:]

with nc.Dataset(master_file, "r") as m_ds:
  lat_key = "lat" if "lat" in m_ds.variables else "latitude"
  lon_key = "lon" if "lon" in m_ds.variables else "longitude"
  grid_lats = m_ds.variables[lat_key][:]
  grid_lons = m_ds.variables[lon_key][:]

valid_mask = (
    (stn_lats_all >= np.min(grid_lats))
    & (stn_lats_all <= np.max(grid_lats))
    & (stn_lons_all >= np.min(grid_lons))
    & (stn_lons_all <= np.max(grid_lons))
)
valid_indices = np.where(valid_mask)[0]
stn_lats = stn_lats_all[valid_indices]
stn_lons = stn_lons_all[valid_indices]

test_lats = stn_lats[g_test]
test_lons = stn_lons[g_test]

df_map = pd.DataFrame({
    "gid": g_test,
    "lat": test_lats,
    "lon": test_lons,
    "True_NO2": y_test,
    "Pred_NO2": y_pred,
})

stn_summary = (
    df_map.groupby("gid")
    .agg({
        "lat": "first",
        "lon": "first",
        "True_NO2": "mean",
        "Pred_NO2": "mean",
    })
    .reset_index()
)

# ==============================================================================
# 4. VECTORIZED INVERSE DISTANCE WEIGHTING (IDW) WITH PLUME DISPERSION
# ==============================================================================
print(
    "⚙️ Executing Inverse Distance Weighting (IDW) for realistic plume"
    " dispersion..."
)
# Define California geographical domain
lon_min, lon_max = -124.5, -114.0
lat_min, lat_max = 32.0, 42.5

grid_lon = np.linspace(lon_min, lon_max, 350)
grid_lat = np.linspace(lat_min, lat_max, 350)
grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)

points = stn_summary[["lon", "lat"]].values
true_vals = stn_summary["True_NO2"].values
pred_vals = stn_summary["Pred_NO2"].values


def compute_idw_surface(x_grid, y_grid, stn_coords, stn_values, power=2.5):
  """Vectorized IDW that mathematically prevents overshoot and mimics air dispersion."""
  dist = np.sqrt(
      (x_grid[:, :, None] - stn_coords[:, 0][None, None, :]) ** 2
      + (y_grid[:, :, None] - stn_coords[:, 1][None, None, :]) ** 2
  )
  dist = np.where(dist < 1e-10, 1e-10, dist)
  weights = 1.0 / (dist**power)
  surface = np.sum(weights * stn_values[None, None, :], axis=2) / np.sum(
      weights, axis=2
  )
  return surface


grid_true = compute_idw_surface(grid_x, grid_y, points, true_vals)
grid_pred = compute_idw_surface(grid_x, grid_y, points, pred_vals)

# ==============================================================================
# 5. FETCH CALIFORNIA STATE & COASTLINE BOUNDARIES (WITH OFFLINE FALLBACK)
# ==============================================================================
print("🗺️ Fetching California geographic borders for overlay...")
ca_geometry = []
try:
  url = (
      "https://raw.githubusercontent.com/johan/world.geo.json/master/countries/USA/CA.geo.json"
  )
  req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
  with urllib.request.urlopen(req, timeout=5) as response:
    geo_data = json.loads(response.read().decode())
    for feature in geo_data.get("features", []):
      geom = feature.get("geometry", {})
      if geom.get("type") == "Polygon":
        ca_geometry.append(np.array(geom["coordinates"][0]))
      elif geom.get("type") == "MultiPolygon":
        for poly in geom["coordinates"]:
          ca_geometry.append(np.array(poly[0]))
  print("✅ Successfully loaded California GeoJSON borders!")
except Exception as e:
  print(
      "⚠️ Could not fetch GeoJSON borders (offline/blocked). Proceeding with"
      " grid overlay."
  )

# ==============================================================================
# 6. RENDER REALISTIC PUBLICATION HEAT MAP
# ==============================================================================
print("🎨 Rendering realistic atmospheric dispersion heat maps...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), sharey=True)

# Strictly clamp color limits to real physical California atmospheric bounds!
vmin, vmax = 3.0, 18.0
cmap = "turbo"  # Modern, high-contrast remote sensing colormap

for ax, grid_data, title, label in [
    (
        ax1,
        grid_true,
        "Observed Ground Telemetry Surface\n(True Climatological Mean NO₂)",
        "EPA Monitoring Sensors",
    ),
    (
        ax2,
        grid_pred,
        (
            "20-Feature XGBoost Digital Twin Surface\n(TEMPO + Weather + GIS"
            " Data Fusion)"
        ),
        f"XGBoost Twin (R²={r2_overall:.2f})",
    ),
]:
  # Plot smooth atmospheric dispersion contours
  c = ax.contourf(
      grid_x,
      grid_y,
      grid_data,
      levels=80,
      cmap=cmap,
      vmin=vmin,
      vmax=vmax,
      extend="both",
      alpha=0.88,
  )

  # Overlay California state boundary lines if downloaded successfully
  for poly in ca_geometry:
    ax.plot(poly[:, 0], poly[:, 1], color="white", linewidth=1.8, zorder=3, alpha=0.9)
    ax.plot(poly[:, 0], poly[:, 1], color="black", linewidth=1.0, zorder=4, alpha=0.8)

  # Overlay physical monitoring stations
  ax.scatter(
      points[:, 0],
      points[:, 1],
      c=stn_summary["True_NO2" if ax == ax1 else "Pred_NO2"],
      cmap=cmap,
      vmin=vmin,
      vmax=vmax,
      edgecolor="black",
      linewidth=1.0,
      s=55,
      zorder=5,
      label=label,
  )

  ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
  ax.set_xlabel("Longitude (°W)", fontsize=12, fontweight="bold")
  ax.set_xlim(lon_min, lon_max)
  ax.set_ylim(lat_min, lat_max)
  ax.grid(True, linestyle="--", alpha=0.3, color="black", zorder=1)
  
  # Removed zorder=6 to prevent standard Matplotlib TypeError!
  ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)

ax1.set_ylabel("Latitude (°N)", fontsize=12, fontweight="bold")

# --- Add City Callouts ---
cities = {
    "Los Angeles Basin": (-118.24, 34.05),
    "Central Valley (Fresno)": (-119.77, 36.75),
    "San Francisco Bay": (-122.42, 37.77),
    "Sacramento": (-121.49, 38.58),
}

for ax in [ax1, ax2]:
  for city_name, (clon, clat) in cities.items():
    ax.annotate(
        city_name,
        xy=(clon, clat),
        xytext=(clon - 1.2, clat + 0.4),
        arrowprops=dict(
            facecolor="white", shrink=0.05, width=1.5, headwidth=6, edgecolor="black"
        ),
        fontsize=9.5,
        fontweight="bold",
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9
        ),
        zorder=6,
    )

# --- Master Colorbar ---
cbar = fig.colorbar(c, ax=[ax1, ax2], orientation="vertical", shrink=0.85, pad=0.02)
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

out_file = OUTPUT_DIR / "california_realistic_pollution_map.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved Realistic California Dispersion Map to: {out_file}")
print("🎯 REALISTIC SPATIAL MAPPING FINISHED!")