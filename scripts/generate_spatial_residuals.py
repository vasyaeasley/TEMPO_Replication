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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Attempt to load contextily for CartoDB background street/coastline tiles
try:
  import contextily as cx

  HAS_CX = True
  print("✅ Contextily GIS library detected! Will render CartoDB Positron tiles.")
except ImportError:
  HAS_CX = False
  print("ℹ️ Contextily not installed. Using fallback vector highway GeoJSON.")

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

print("🗺️ STARTING SPATIAL RESIDUAL MAPPING DIAGNOSTIC (LAYOUT OPTIMIZED) 🗺️")
print("=" * 75)

if not data_file.exists():
  raise FileNotFoundError(f"Could not locate 20-feature dataset at: {data_file}")

# ==============================================================================
# 2. LOAD DATA & TRAIN OPTIMAL XGBOOST DIGITAL TWIN
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
r2_val = r2_score(y_test, y_pred)
rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
mae_val = mean_absolute_error(y_test, y_pred)

# Calculate exact mathematical residuals: True - Predicted
residuals = y_test - y_pred
mean_global_bias = np.mean(residuals)

print(f"📊 Global Unseen Test Metrics (N={len(y_test):,}):")
print(f"   * R² Score:         {r2_val:.3f}")
print(f"   * RMSE:             {rmse_val:.2f} ppb")
print(f"   * MAE:              {mae_val:.2f} ppb")
print(f"   * Global Mean Bias: {mean_global_bias:+.3f} ppb")

# ==============================================================================
# 3. EXTRACT PHYSICAL STATION COORDINATES & AGGREGATE ERRORS
# ==============================================================================
print("🔍 Extracting physical station coordinates and calculating site errors...")
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

df_res = pd.DataFrame({
    "gid": g_test,
    "lat": test_lats,
    "lon": test_lons,
    "True_NO2": y_test,
    "Pred_NO2": y_pred,
    "Residual": residuals,
    "Abs_Residual": np.abs(residuals),
})

stn_summary = (
    df_res.groupby("gid")
    .agg({
        "lat": "first",
        "lon": "first",
        "True_NO2": "mean",
        "Pred_NO2": "mean",
        "Residual": "mean",
        "Abs_Residual": "mean",
        "gid": "count",
    })
    .rename(columns={"gid": "sample_count"})
    .reset_index()
)

print(
    f"📋 Aggregated error metrics across {len(stn_summary)} unique EPA AirNow"
    " monitoring stations."
)

# ==============================================================================
# 4. LOAD LOCAL INTERSTATE HIGHWAY CACHE
# ==============================================================================
print("🛣️ Loading US Interstate GeoJSON from local disk cache...")
highway_lines = []
cache_file = PROCESSED_DIR / "us_interstates_cache.json"

try:
  if cache_file.exists():
    with open(cache_file, "r") as f:
      road_data = json.load(f)
  else:
    url_roads = "https://raw.githubusercontent.com/giswqs/geodata/main/GeoJSON/us_interstates.geojson"
    req = urllib.request.Request(
        url_roads, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
      road_data = json.loads(response.read().decode())
    with open(cache_file, "w") as f:
      json.dump(road_data, f)

  for feature in road_data.get("features", []):
    geom = feature.get("geometry", {})
    if geom.get("type") == "LineString":
      highway_lines.append(np.array(geom["coordinates"]))
    elif geom.get("type") == "MultiLineString":
      for line in geom["coordinates"]:
        highway_lines.append(np.array(line))
  print(f"✅ Successfully loaded {len(highway_lines)} highway segments!")
except Exception as e:
  print(f"⚠️ Could not load highway GeoJSON ({e}). Proceeding with street tiles.")

ca_geometry = []
if not HAS_CX:
  try:
    url_ca = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries/USA/CA.geo.json"
    req = urllib.request.Request(url_ca, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as response:
      geo_data = json.loads(response.read().decode())
      for feature in geo_data.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") == "Polygon":
          ca_geometry.append(np.array(geom["coordinates"][0]))
        elif geom.get("type") == "MultiPolygon":
          for poly in geom["coordinates"]:
            ca_geometry.append(np.array(poly[0]))
  except Exception:
    pass

# ==============================================================================
# 5. RENDER PUBLICATION-GRADE 2-PANEL RESIDUAL MAP (COLLISION FIXED)
# ==============================================================================
print("🎨 Rendering 2-Panel Spatial Residual Map...")
# Using constrained_layout=True mathematically prevents colorbar overlapping!
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(18.0, 8.5), constrained_layout=True
)

cmap = "coolwarm"
vmin, vmax = -3.5, 3.5
norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

# --- PANEL 1: FULL CALIFORNIA STATE DOMAIN ---
ax1.set_xlim(-124.6, -113.8)
ax1.set_ylim(32.1, 42.2)

# --- PANEL 2: ZOOMED LOS ANGELES BASIN INVERSION CORRIDOR ---
ax2.set_xlim(-118.90, -117.00)
ax2.set_ylim(33.60, 34.30)

for ax in [ax1, ax2]:
  ax.set_facecolor("#e8f4f8")

  if HAS_CX:
    try:
      cx.add_basemap(
          ax,
          crs=4326,
          source=cx.providers.CartoDB.Positron,
          zoom=8 if ax == ax1 else 10,
          attribution="",
      )
    except Exception as e:
      pass

  if not HAS_CX:
    for poly in ca_geometry:
      ax.fill(poly[:, 0], poly[:, 1], color="#fdfdfd", zorder=1)
      ax.plot(poly[:, 0], poly[:, 1], color="#333333", linewidth=1.5, zorder=2)

  for line in highway_lines:
    xl, yl = ax.get_xlim(), ax.get_ylim()
    if (
        np.min(line[:, 0]) <= xl[1]
        and np.max(line[:, 0]) >= xl[0]
        and np.min(line[:, 1]) <= yl[1]
        and np.max(line[:, 1]) >= yl[0]
    ):
      ax.plot(
          line[:, 0],
          line[:, 1],
          color="#64748b",
          linewidth=1.5 if ax == ax1 else 2.2,
          alpha=0.85,
          zorder=3,
      )

  sc = ax.scatter(
      stn_summary["lon"],
      stn_summary["lat"],
      c=stn_summary["Residual"],
      cmap=cmap,
      norm=norm,
      edgecolor="black",
      linewidth=1.3,
      s=110 if ax == ax1 else 180,
      alpha=0.90,
      zorder=5,
  )

  ax.set_aspect("auto", adjustable="box")
  ax.grid(True, linestyle="--", alpha=0.4, color="#94a3b8", zorder=0)
  ax.set_xlabel("Longitude (°W)", fontsize=12, fontweight="bold")
  ax.tick_params(axis="both", labelsize=11)
  for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")

ax1.set_ylabel("Latitude (°N)", fontsize=12, fontweight="bold")
ax1.set_title(
    "State-Wide Mean Residual Distribution\n(California Domain, N=47,956"
    " hours)",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
ax2.set_title(
    "Micro-Scale Urban Inversion Residual Tracking\n(Los Angeles Basin"
    " Freight Corridors)",
    fontsize=14,
    fontweight="bold",
    pad=12,
)

# --- Add Clean Diagnostic Landmark Callouts to LA Basin ---
la_landmarks = {
    "Downtown LA\n(I-10 Hub)": (-118.24, 34.05),
    "Pomona Valley\n(Inversion Zone)": (-117.75, 34.06),
    "Riverside / Inland": (-117.37, 33.98),
    "Anaheim\n(I-5 Corridor)": (-117.91, 33.83),
}

for name, (clon, clat) in la_landmarks.items():
  ax2.annotate(
      name,
      xy=(clon, clat),
      xytext=(clon - 0.22, clat + 0.08),
      arrowprops=dict(
          facecolor="#111827",
          shrink=0.08,
          width=1.5,
          headwidth=5,
          edgecolor="none",
      ),
      fontsize=9.5,
      fontweight="bold",
      color="#111827",
      bbox=dict(
          boxstyle="round,pad=0.3",
          facecolor="white",
          edgecolor="#4b5563",
          alpha=0.92,
      ),
      zorder=6,
  )

# Master Diverging Colorbar attached strictly to the right axis!
cbar = fig.colorbar(
    sc, ax=ax2, orientation="vertical", shrink=0.88, pad=0.04
)
cbar.set_label(
    "Mean Site Residual Error (True - Predicted, ppb)",
    fontsize=13,
    fontweight="bold",
)
cbar.ax.tick_params(labelsize=11)

stats_text = (
    f"Model: 20-Feature XGBoost\n"
    f"Global Test R²:    {r2_val:.3f}\n"
    f"Global Test RMSE:  {rmse_val:.2f} ppb\n"
    f"Mean Site Bias:    {mean_global_bias:+.2f} ppb\n"
    f"Stations Evaluated: {len(stn_summary)}"
)

ax1.text(
    0.04,
    0.96,
    stats_text,
    transform=ax1.transAxes,
    fontsize=11.5,
    fontweight="bold",
    family="monospace",
    verticalalignment="top",
    bbox=dict(
        boxstyle="round,pad=0.6",
        facecolor="white",
        edgecolor="#4b5563",
        alpha=0.94,
        linewidth=1.5,
    ),
    zorder=10,
)

plt.suptitle(
    "XGBoost Digital Twin Spatial Residual Diagnostic (True - Predicted)\n"
    "Evaluating Geographic Generalization and Micro-Scale Urban Bias",
    fontsize=16,
    fontweight="bold",
)

out_file = OUTPUT_DIR / "tempo_spatial_residual_map.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved Publication Spatial Residual Map to: {out_file}")
print("🎯 SPATIAL RESIDUAL MAPPING FINISHED!")