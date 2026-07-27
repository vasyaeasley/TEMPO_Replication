import os

# Thread safety controls
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import netCDF4 as nc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
data_file = PROCESSED_DIR / "epa_point_dataset_14months_20features.npz"
airnow_dir = BASE_DIR / "data" / "raw" / "epa" / "AirNow"
if not airnow_dir.exists():
  airnow_dir = BASE_DIR / "data" / "raw" / "AirNow"

print("🔍 DIAGNOSING THE TRUE WORST POLLUTION ANOMALY 🔍")
print("=" * 75)

data = np.load(data_file, allow_pickle=True)
X_full = np.vstack([data["X_train"], data["X_test"]])
y_full = np.concatenate([data["y_train"], data["y_test"]])
g_full = np.concatenate([data["groups_train"], data["groups_test"]])

X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X_full, y_full, g_full, test_size=0.20, random_state=42
)

feature_names = np.array([
    "TEMPO_NO2",
    "blh",
    "traffic",
    "t2m",
    "elev",
    "pop",
    "month_cos",
    "road_density",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "sp",
    "day_of_week_cos",
    "month_sin",
    "is_weekend",
    "u10",
    "v10",
    "d2m",
    "tcc",
    "solar_zenith_angle",
])
if "feature_names" in data:
  feature_names = np.array([str(f) for f in data["feature_names"]])

# Train baseline model to get exact prediction for the worst sample
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
y_pred = best_xgb.predict(X_test)

# 1. Locate the single highest true NO2 observation in the entire test set!
max_idx = np.argmax(y_test)
true_max_no2 = y_test[max_idx]
pred_max_no2 = y_pred[max_idx]
target_gid = g_test[max_idx]

# Extract coordinates of this station from AirNow
airnow_files = sorted(list(airnow_dir.glob("*.nc")))
with nc.Dataset(airnow_files[0], "r") as a_ds:
  stn_lats_all = a_ds.variables["latitude"][:]
  stn_lons_all = a_ds.variables["longitude"][:]

master_file = PROCESSED_DIR / "era5_california_1x1km_master.nc"
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

stn_lat = stn_lats[target_gid]
stn_lon = stn_lons[target_gid]

# Decode approximate time/season from trigonometric features
h_sin = X_test[max_idx, np.where(feature_names == "hour_sin")[0][0]]
h_cos = X_test[max_idx, np.where(feature_names == "hour_cos")[0][0]]
m_sin = X_test[max_idx, np.where(feature_names == "month_sin")[0][0]]
m_cos = X_test[max_idx, np.where(feature_names == "month_cos")[0][0]]

utc_hr = int(np.round(np.mod(np.arctan2(h_sin, h_cos) * 24 / (2 * np.pi), 24)))
month_est = int(
    np.round(np.mod(np.arctan2(m_sin, m_cos) * 12 / (2 * np.pi), 12))
)
month_est = 12 if month_est == 0 else month_est
local_hr = (utc_hr - 7) % 24

print(f"🚨 SINGLE WORST POLLUTION HOUR IN TEST SET:")
print(f"   * Station Group ID (gid): {target_gid}")
print(
    f"   * Coordinates:            {stn_lat:.4f}°N, {stn_lon:.4f}°W"
    " (Search on Google Maps!)"
)
print(
    f"   * Approximate Timing:     Month ~{month_est}, Local Time ~{local_hr:02d}:00"
    " PDT"
)
print(f"   * True EPA NO₂ Reading:   {true_max_no2:.2f} ppb")
print(f"   * XGBoost Prediction:     {pred_max_no2:.2f} ppb")
print(f"   * Instantaneous Error:    {pred_max_no2 - true_max_no2:+.2f} ppb")

print("\n📋 FEATURE VALUES FOR THIS EXACT EXTREME HOUR:")
print("-" * 55)
for name, val in zip(feature_names, X_test[max_idx]):
  print(f"   * {name:<20}: {val:+.4f}")
print("-" * 55)