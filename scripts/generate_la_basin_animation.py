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

# Attempt to load contextily for CartoDB background street/coastline tiles
try:
  import contextily as cx

  HAS_CX = True
  print(
      "✅ Contextily GIS library detected! Will render CartoDB Positron road"
      " tiles."
  )
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

print("🎬 STARTING WIDESCREEN LA BASIN GIS DIURNAL ANIMATION 🎬")
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
print(f"🎯 State-Wide Unseen Test Set R²: {r2_overall:.3f}")

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

# AGGRESSIVE WIDESCREEN BOUNDING BOX: Expanded horizontally from Malibu out to Redlands!
lon_min, lon_max = -118.90, -117.00
lat_min, lat_max = 33.60, 34.30

active_hours = sorted([
    h for h in df_anim["Hour"].unique() if len(df_anim[df_anim["Hour"] == h]) > 50
])
print(
    f"📋 Discovered {len(active_hours)} active daylight hours for animation:"
    f" {active_hours}"
)

# ==============================================================================
# 4. LOCAL FILE CACHING FOR INTERSTATE HIGHWAYS (PREVENTS TIMEOUT DROPS!)
# ==============================================================================
print(
    "🛣️ Loading US Interstate GeoJSON (with local disk caching and 30s"
    " timeout)..."
)
highway_lines = []
cache_file = PROCESSED_DIR / "us_interstates_cache.json"

try:
  if cache_file.exists():
    print(f"📦 Loading highways directly from local disk cache: {cache_file}")
    with open(cache_file, "r") as f:
      road_data = json.load(f)
  else:
    print("🌐 Downloading highways from GitHub (30-second timeout allowance)...")
    url_roads = "https://raw.githubusercontent.com/giswqs/geodata/main/GeoJSON/us_interstates.geojson"
    req = urllib.request.Request(
        url_roads, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=30) as response:
      road_data = json.loads(response.read().decode())
    # Save to disk so we never have to download it again!
    with open(cache_file, "w") as f:
      json.dump(road_data, f)
    print("✅ Successfully cached highway GeoJSON to local disk!")

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
# 5. BUILD MATPLOTLIB WIDESCREEN GIS ANIMATION LOOP
# ==============================================================================
print("⚙️ Initializing widescreen LA Basin GIS figure and rendering frames...")
# 20.0 x 8.0 inches enforces a dramatic, widescreen 2.5:1 cinematic ratio!
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20.0, 8.0), sharey=True)

vmin, vmax = 3.0, 24.0
cmap = "turbo"
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# Pre-draw static cartographic background & road grids
for ax in [ax1, ax2]:
  ax.set_facecolor("#e8f4f8")
  ax.set_xlim(lon_min, lon_max)
  ax.set_ylim(lat_min, lat_max)

  # 1. Overlay CartoDB Positron High-Resolution City/Road Tiles
  if HAS_CX:
    try:
      cx.add_basemap(
          ax,
          crs=4326,
          source=cx.providers.CartoDB.Positron,
          zoom=10,
          attribution="",
      )
    except Exception as e:
      print(f"⚠️ Could not fetch CartoDB tiles ({e}). Using vector roads.")

  # 2. Overlay Vector State Coastlines if Contextily was offline
  if not HAS_CX:
    for poly in ca_geometry:
      ax.fill(poly[:, 0], poly[:, 1], color="#fdfdfd", zorder=1)
      ax.plot(poly[:, 0], poly[:, 1], color="#333333", linewidth=1.8, zorder=2)

  # 3. Overlay Major Interstate Highway Lines
  for line in highway_lines:
    if (
        np.min(line[:, 0]) <= lon_max
        and np.max(line[:, 0]) >= lon_min
        and np.min(line[:, 1]) <= lat_max
        and np.max(line[:, 1]) >= lat_min
    ):
      ax.plot(
          line[:, 0],
          line[:, 1],
          color="#64748b",
          linewidth=2.0,
          alpha=0.90,
          zorder=3,
      )

  # CRITICAL WIDESCREEN FIX: Must be called AFTER cx.add_basemap() to break aspect lock!
  ax.set_aspect("auto", adjustable="box")
  ax.grid(True, linestyle="--", alpha=0.4, color="#94a3b8", zorder=0)
  ax.set_xlabel("Longitude (°W)", fontsize=12, fontweight="bold")

ax1.set_ylabel("Latitude (°N)", fontsize=12, fontweight="bold")

# --- INITIALIZE WITH FRAME 0 DATA ---
first_hr_df = df_anim[df_anim["Hour"] == active_hours[0]]
stn_init = (
    first_hr_df.groupby("gid")
    .agg({
        "lat": "first",
        "lon": "first",
        "True_NO2": "mean",
        "Pred_NO2": "mean",
    })
    .reset_index()
)
init_coords = np.column_stack([stn_init["lon"], stn_init["lat"]])

sc1 = ax1.scatter(
    init_coords[:, 0],
    init_coords[:, 1],
    c=stn_init["True_NO2"].values,
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    linewidth=1.4,
    s=170,
    zorder=5,
)
sc2 = ax2.scatter(
    init_coords[:, 0],
    init_coords[:, 1],
    c=stn_init["Pred_NO2"].values,
    cmap=cmap,
    norm=norm,
    edgecolor="black",
    linewidth=1.4,
    s=170,
    zorder=5,
)

# Master Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(
    sm, ax=[ax1, ax2], orientation="vertical", shrink=0.85, pad=0.02
)
cbar.set_label(
    "Mean Ground-Level NO₂ Concentration (ppb)", fontsize=13, fontweight="bold"
)

# --- Clean Localized LA Basin City Callouts ---
la_landmarks = {
    "Downtown LA\n(I-10 / I-110 Hub)": (-118.24, 34.05),
    "Pomona Valley\n(Inversion Zone)": (-117.75, 34.06),
    "Riverside / Inland": (-117.37, 33.98),
    "Anaheim\n(I-5 Corridor)": (-117.91, 33.83),
    "San Fernando Valley": (-118.45, 34.22),
}

for ax in [ax1, ax2]:
  for name, (clon, clat) in la_landmarks.items():
    ax.annotate(
        name,
        xy=(clon, clat),
        xytext=(clon - 0.22, clat + 0.10),
        arrowprops=dict(
            facecolor="#111827",
            shrink=0.08,
            width=1.5,
            headwidth=5,
            edgecolor="none",
        ),
        fontsize=9,
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

title_sup = fig.suptitle("", fontsize=17, fontweight="bold", y=0.98)


def update_frame(frame_idx):
  """Updates sensor markers and calculates localized LA Basin R² for each hour."""
  current_hour = active_hours[frame_idx]
  sub = df_anim[df_anim["Hour"] == current_hour]

  la_mask = (
      (sub["lon"] >= lon_min)
      & (sub["lon"] <= lon_max)
      & (sub["lat"] >= lat_min)
      & (sub["lat"] <= lat_max)
  )
  sub_la = sub[la_mask]

  stn_hr = (
      sub.groupby("gid")
      .agg({
          "lat": "first",
          "lon": "first",
          "True_NO2": "mean",
          "Pred_NO2": "mean",
      })
      .reset_index()
  )

  if len(sub_la) > 10:
    la_r2 = r2_score(sub_la["True_NO2"], sub_la["Pred_NO2"])
  else:
    la_r2 = 0.0

  coords = np.column_stack([stn_hr["lon"], stn_hr["lat"]])
  sc1.set_offsets(coords)
  sc1.set_array(stn_hr["True_NO2"].values)
  sc1.set_clim(vmin, vmax)

  sc2.set_offsets(coords)
  sc2.set_array(stn_hr["Pred_NO2"].values)
  sc2.set_clim(vmin, vmax)

  ax1.set_title(
      f"Observed EPA Network (LA Basin)\nTime: {current_hour:02d}:00 UTC",
      fontsize=14,
      fontweight="bold",
      pad=12,
  )
  ax2.set_title(
      f"XGBoost Twin Prediction\nTime: {current_hour:02d}:00 UTC (LA Basin"
      f" R²={la_r2:.2f})",
      fontsize=14,
      fontweight="bold",
      pad=12,
  )
  title_sup.set_text(
      "Los Angeles Basin High-Resolution GIS Diurnal Tracking — Frame"
      f" {frame_idx + 1}/{len(active_hours)}"
  )

  return sc1, sc2, ax1, ax2, title_sup


print(
    "⏳ Compiling widescreen LA Basin GIS animation frames into GIF (takes"
    " ~15-25 seconds)..."
)
anim = animation.FuncAnimation(
    fig, update_frame, frames=len(active_hours), interval=850, blit=False
)

out_file = OUTPUT_DIR / "la_basin_diurnal_twin_animation.gif"
anim.save(out_file, writer="pillow", fps=1.2, dpi=150)
plt.close(fig)

print(f"✅ Saved Widescreen LA Basin Animated GIS GIF to: {out_file}")
print("🎯 LA BASIN GIS ANIMATION FINISHED!")