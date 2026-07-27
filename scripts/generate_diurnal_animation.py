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
import matplotlib.animation as animation
import matplotlib.colors as mcolors
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

print("🎬 STARTING TEMPO DIURNAL DIGITAL TWIN ANIMATION GENERATION 🎬")
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
# 3. EXTRACT PHYSICAL STATION COORDINATES & TIMESTAMPS
# ==============================================================================
print("🔍 Extracting physical station coordinates and local hours...")
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

# Decode trigonometric hour variables (index 8: hour_sin, index 9: hour_cos)
hour_sin = X_test[:, 8]
hour_cos = X_test[:, 9]
hours_raw = (
    np.round(np.arctan2(hour_sin, hour_cos) * 24.0 / (2 * np.pi)) % 24
).astype(int)

df_anim = pd.DataFrame({
    "gid": g_test,
    "lat": test_lats,
    "lon": test_lons,
    "Hour": hours_raw,
    "True_NO2": y_test,
    "Pred_NO2": y_pred,
})

# Filter for active daylight observation hours with rich station coverage
active_hours = sorted([
    h for h in df_anim["Hour"].unique() if len(df_anim[df_anim["Hour"] == h]) > 50
])
print(f"📋 Discovered {len(active_hours)} active daylight hours for animation: {active_hours}")

# ==============================================================================
# 4. FETCH CALIFORNIA BORDERS & COASTLINES
# ==============================================================================
print("🗺️ Fetching California geographic borders...")
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
  print("⚠️ Could not fetch GeoJSON borders. Using standard coordinate box.")

# ==============================================================================
# 5. BUILD MATPLOTLIB ANIMATION LOOP WITH LOCKED NORMALIZATION
# ==============================================================================
print("⚙️ Initializing animation figure and rendering frames...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), sharey=True)

lon_min, lon_max = -124.5, -114.0
lat_min, lat_max = 32.0, 42.5
vmin, vmax = 2.0, 20.0
cmap = "turbo"

# Lock normalization explicitly so Matplotlib CANNOT default to 0.0 - 1.0!
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# Pre-draw static background geometry
for ax in [ax1, ax2]:
  ax.set_facecolor("#eef2f5")
  for poly in ca_geometry:
    ax.fill(poly[:, 0], poly[:, 1], color="#fdfdfd", zorder=1)
    ax.plot(poly[:, 0], poly[:, 1], color="#333333", linewidth=1.5, zorder=2)
  ax.set_xlim(lon_min, lon_max)
  ax.set_ylim(lat_min, lat_max)
  ax.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1", zorder=0)
  ax.set_xlabel("Longitude (°W)", fontsize=12, fontweight="bold")

ax1.set_ylabel("Latitude (°N)", fontsize=12, fontweight="bold")

# Initialize empty scatter plots using explicit norm
sc1 = ax1.scatter([], [], cmap=cmap, norm=norm, edgecolor="black", linewidth=1.2, s=110, zorder=4)
sc2 = ax2.scatter([], [], cmap=cmap, norm=norm, edgecolor="black", linewidth=1.2, s=110, zorder=4)

# Build Colorbar from a standalone ScalarMappable to guarantee 2.0 to 20.0 limits!
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation="vertical", shrink=0.85, pad=0.02)
cbar.set_label("Mean Ground-Level NO₂ Concentration (ppb)", fontsize=13, fontweight="bold")

# Add City Annotations
cities = {
    "Los Angeles Basin": (-118.24, 34.05),
    "Central Valley": (-119.77, 36.75),
    "Bay Area": (-122.42, 37.77),
}
for ax in [ax1, ax2]:
  for city_name, (clon, clat) in cities.items():
    ax.annotate(
        city_name,
        xy=(clon, clat), xytext=(clon - 1.2, clat + 0.35),
        arrowprops=dict(facecolor="#333333", shrink=0.08, width=1.5, headwidth=5, edgecolor="none"),
        fontsize=9, fontweight="bold", color="#111827",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#6b7280", alpha=0.9),
        zorder=5,
    )

title_sup = fig.suptitle("", fontsize=17, fontweight="bold", y=0.98)


def update_frame(frame_idx):
  """Updates scatter points and titles for each daylight observation hour."""
  current_hour = active_hours[frame_idx]
  sub = df_anim[df_anim["Hour"] == current_hour]

  # Aggregate station means for the current hour
  stn_hr = sub.groupby("gid").agg({
      "lat": "first", "lon": "first",
      "True_NO2": "mean", "Pred_NO2": "mean"
  }).reset_index()

  if len(sub) > 10:
    hr_r2 = r2_score(sub["True_NO2"], sub["Pred_NO2"])
  else:
    hr_r2 = 0.0

  # Update coordinates and color arrays
  coords = np.column_stack([stn_hr["lon"], stn_hr["lat"]])
  sc1.set_offsets(coords)
  sc1.set_array(stn_hr["True_NO2"].values)
  sc1.set_clim(vmin, vmax)  # Force clamp limits on every frame!

  sc2.set_offsets(coords)
  sc2.set_array(stn_hr["Pred_NO2"].values)
  sc2.set_clim(vmin, vmax)  # Force clamp limits on every frame!

  # Update dynamic titles
  ax1.set_title(f"Observed EPA Ground Network\nTime: {current_hour:02d}:00 UTC", fontsize=14, fontweight="bold", pad=12)
  ax2.set_title(f"XGBoost Digital Twin Prediction\nTime: {current_hour:02d}:00 UTC (Hourly R²={hr_r2:.2f})", fontsize=14, fontweight="bold", pad=12)
  title_sup.set_text(f"TEMPO Geostationary Digital Twin Diurnal Tracking — Frame {frame_idx + 1}/{len(active_hours)}")

  return sc1, sc2, ax1, ax2, title_sup


print("⏳ Compiling animation frames into GIF (this takes ~15-25 seconds)...")
anim = animation.FuncAnimation(fig, update_frame, frames=len(active_hours), interval=800, blit=False)

out_file = OUTPUT_DIR / "california_diurnal_twin_animation.gif"
anim.save(out_file, writer="pillow", fps=1.2, dpi=150)
plt.close(fig)

print(f"✅ Saved Animated Diurnal GIF to: {out_file}")
print("🎯 ANIMATION FINISHED!")