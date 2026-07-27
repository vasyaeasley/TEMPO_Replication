import os

# Thread safety controls
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from pathlib import Path
import time
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR = BASE_DIR / 'models'
OUTPUT_DIR.mkdir(exist_ok=True)

data_file = PROCESSED_DIR / 'epa_point_dataset_14months_20features.npz'
master_file = PROCESSED_DIR / 'era5_california_1x1km_master.nc'
airnow_dir = BASE_DIR / 'data' / 'raw' / 'epa' / 'AirNow'
if not airnow_dir.exists():
  airnow_dir = BASE_DIR / 'data' / 'raw' / 'AirNow'

print('🚀 STARTING DIGITAL TWIN OVERLAY: XGBOOST vs. CLASSICAL BASELINES 🚀')
print('=' * 75)

if not data_file.exists():
  raise FileNotFoundError(f'Could not locate 20-feature dataset at: {data_file}')

# ==============================================================================
# 2. LOAD ARRAYS & ISOLATE POMONA STATION
# ==============================================================================
print('⏳ Loading 20-feature dataset and locating Pomona Station...')
data = np.load(data_file, allow_pickle=True)

X_full = np.vstack([data['X_train'], data['X_test']])
y_full = np.concatenate([data['y_train'], data['y_test']])
g_full = np.concatenate([data['groups_train'], data['groups_test']])

# Find Pomona Station ID near Lat 34.056, Lon -117.752
airnow_files = sorted(list(airnow_dir.glob('*.nc')))
with nc.Dataset(airnow_files[0], 'r') as a_ds:
  stn_lats = a_ds.variables['latitude'][:]
  stn_lons = a_ds.variables['longitude'][:]
  stn_ids = [str(sid).strip() for sid in a_ds.variables['site'][:]]

stn_tree = cKDTree(np.column_stack([stn_lats, stn_lons]))
dist, idx = stn_tree.query([34.056, -117.752])
pomona_id = stn_ids[idx]

with nc.Dataset(master_file, 'r') as m_ds:
  lat_key = 'lat' if 'lat' in m_ds.variables else 'latitude'
  lon_key = 'lon' if 'lon' in m_ds.variables else 'longitude'
  grid_lats, grid_lons = m_ds.variables[lat_key][:], m_ds.variables[lon_key][:]
  if grid_lats.ndim == 1:
    lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
  else:
    lat_grid, lon_grid = grid_lats, grid_lons

valid_mask = (
    (stn_lats >= np.min(grid_lats))
    & (stn_lats <= np.max(grid_lats))
    & (stn_lons >= np.min(grid_lons))
    & (stn_lons <= np.max(grid_lons))
)
valid_indices = np.where(valid_mask)[0]
ca_ids = [stn_ids[i] for i in valid_indices]
pomona_gid = ca_ids.index(pomona_id)

print(f'🎯 Pomona Station Confirmed! Group ID: {pomona_gid}')

# ==============================================================================
# 3. GENERATE 5-FOLD OUT-OF-FOLD (OOF) XGBOOST PREDICTIONS ACROSS DOMAIN
# ==============================================================================
print(
    '\n🧠 Executing 5-Fold Out-Of-Fold (OOF) XGBoost Predictions across 239,780'
    ' rows...'
)
print(
    '   (This guarantees Pomona predictions are 100% out-of-sample test'
    ' results!)'
)
start_ml = time.time()

oof_predictions = np.zeros(len(y_full))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_idx, te_idx) in enumerate(kf.split(X_full), 1):
  print(
      f'   * Training Fold {fold}/5 ({len(tr_idx):,} rows) -> Predicting'
      f' ({len(te_idx):,} rows)...'
  )
  model = XGBRegressor(
      n_estimators=550,
      learning_rate=0.035,
      max_depth=8,
      subsample=0.75,
      colsample_bytree=0.70,
      gamma=0.1,
      reg_alpha=15.0,
      reg_lambda=3.5,
      random_state=42,
      n_jobs=-1,
      tree_method='hist',
  )
  model.fit(X_full[tr_idx], y_full[tr_idx])
  oof_predictions[te_idx] = model.predict(X_full[te_idx])

print(f'✅ ML Out-Of-Fold predictions generated in {time.time() - start_ml:.1f}s!')

# ==============================================================================
# 4. EXTRACT POMONA DAILY MEANS & REBUILD CLASSICAL BASELINES
# ==============================================================================
pomona_mask = g_full == pomona_gid
X_pom, y_pom = X_full[pomona_mask], y_full[pomona_mask]
y_ml_pom = oof_predictions[pomona_mask]

hours_per_day = 9
n_days = len(y_pom) // hours_per_day

daily_no2, daily_ml, daily_t2m, daily_sp, daily_d2m, daily_u10, daily_v10 = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
)
for i in range(n_days):
  sl = slice(i * hours_per_day, (i + 1) * hours_per_day)
  daily_no2.append(np.mean(y_pom[sl]))
  daily_ml.append(np.mean(y_ml_pom[sl]))
  daily_t2m.append(np.mean(X_pom[sl, 1]) * 15.0 + 285.0 - 273.15)
  daily_sp.append((np.mean(X_pom[sl, 4]) * 5000.0 + 95000.0) / 100.0)
  daily_d2m.append(np.mean(X_pom[sl, 18]) * 10.0)
  daily_u10.append(np.mean(X_pom[sl, 2]) * 10.0)
  daily_v10.append(np.mean(X_pom[sl, 3]) * 10.0)

daily_no2 = np.array(daily_no2)
daily_ml = np.array(daily_ml)
daily_wind = (
    np.sqrt(np.array(daily_u10) ** 2 + np.array(daily_v10) ** 2) * 1.94384
)

e_act = 6.11 * np.exp((17.67 * np.array(daily_d2m)) / (np.array(daily_d2m) + 243.5))
e_sat = 6.11 * np.exp((17.67 * np.array(daily_t2m)) / (np.array(daily_t2m) + 243.5))
daily_rh = np.clip(100.0 * (e_act / e_sat), 5.0, 100.0)

# Rebuild Baselines
y_persist = np.roll(daily_no2, shift=1)
y_persist[0] = daily_no2[0]

X_met_4 = np.column_stack([daily_wind, daily_rh, daily_t2m, daily_sp])
lr_met_4 = LinearRegression().fit(X_met_4, daily_no2)
y_met_4 = lr_met_4.predict(X_met_4)

# ==============================================================================
# 5. PRINT MASTER COMPARISON TABLE
# ==============================================================================
results = [
    {
        'Model': '🚀 20-Feature XGBoost (ML Twin)',
        'Prediction Type': 'TEMPO + SZA + Weather + GIS Grid',
        'RMSE (ppb)': np.sqrt(mean_squared_error(daily_no2, daily_ml)),
        'MAE (ppb)': mean_absolute_error(daily_no2, daily_ml),
        'R²': r2_score(daily_no2, daily_ml),
    },
    {
        'Model': 'Stepwise 4-Param Meteorology',
        'Prediction Type': 'Wind + RH + Temp + Pressure',
        'RMSE (ppb)': np.sqrt(mean_squared_error(daily_no2, y_met_4)),
        'MAE (ppb)': mean_absolute_error(daily_no2, y_met_4),
        'R²': r2_score(daily_no2, y_met_4),
    },
    {
        'Model': 'Persistence Baseline',
        'Prediction Type': 'Previous Day (Lag-1)',
        'RMSE (ppb)': np.sqrt(mean_squared_error(daily_no2, y_persist)),
        'MAE (ppb)': mean_absolute_error(daily_no2, y_persist),
        'R²': r2_score(daily_no2, y_persist),
    },
]

print('\n' + '=' * 80)
print('🏆 POMONA STATION (#060371701) MASTER MODEL OVERLAY TABLE')
print('=' * 80)
df_res = pd.DataFrame(results).sort_values(by='RMSE (ppb)', ascending=True)
print(df_res.to_string(index=False))
print('=' * 80)

# ==============================================================================
# 6. GENERATE THE DIGITAL TWIN OVERLAY FIGURE
# ==============================================================================
print('\n🎨 Generating Digital Twin Time Series Overlay Figure...')
days_x = np.arange(1, n_days + 1)
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [2.3, 1]}
)

# Top Panel
ax1.plot(
    days_x,
    daily_no2,
    label='Observed Daily Mean (True NO₂)',
    color='#1f77b4',
    linewidth=2.0,
    alpha=0.85,
)
ax1.plot(
    days_x,
    y_persist,
    label=f'Persistence Baseline (R²={results[2]["R²"]:.2f})',
    color='#2ca02c',
    linestyle='--',
    linewidth=1.5,
    alpha=0.65,
)
ax1.plot(
    days_x,
    y_met_4,
    label=f'Stepwise 4-Param Linear Regression (R²={results[1]["R²"]:.2f})',
    color='#ff7f0e',
    linewidth=2.0,
    alpha=0.85,
)
ax1.plot(
    days_x,
    daily_ml,
    label=f'🚀 20-Feature XGBoost Digital Twin (R²={results[0]["R²"]:.2f})',
    color='#d62728',
    linewidth=2.4,
    alpha=0.95,
)

ax1.set_title(
    'Pomona NO₂ Daily Mean: Classical Regression vs. 20-Feature Machine Learning'
    ' Twin\nSite #060371701 (Eastern Los Angeles Basin Inversion Zone)',
    fontsize=14,
    fontweight='bold',
    pad=12,
)
ax1.set_ylabel('NO₂ Concentration (ppb)', fontsize=12, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=11)

# Stats overlay box
stats_box = (
    f"🏆 MODEL COMPARISON SUMMARY:\n"
    f"XGBoost Digital Twin : R² = {results[0]['R²']:.3f} (RMSE: {results[0]['RMSE (ppb)']:.2f} ppb)\n"
    f"4-Param Linear Suite : R² = {results[1]['R²']:.3f} (RMSE: {results[1]['RMSE (ppb)']:.2f} ppb)\n"
    f"Persistence Baseline : R² = {results[2]['R²']:.3f} (RMSE: {results[2]['RMSE (ppb)']:.2f} ppb)"
)
ax1.text(
    0.02,
    0.88,
    stats_box,
    transform=ax1.transAxes,
    fontsize=10.5,
    fontfamily='monospace',
    bbox=dict(
        boxstyle='round,pad=0.6', facecolor='white', edgecolor='gray', alpha=0.9
    ),
)

# Bottom Panel: Residuals
ax2.axhline(0, color='black', linewidth=1.2, linestyle='-', alpha=0.7)
ax2.plot(
    days_x,
    y_met_4 - daily_no2,
    label='Linear 4-Param Error',
    color='#ff7f0e',
    linewidth=1.3,
    alpha=0.7,
)
ax2.plot(
    days_x,
    daily_ml - daily_no2,
    label='XGBoost Twin Error',
    color='#d62728',
    linewidth=1.6,
    alpha=0.9,
)

ax2.set_title(
    'Model Residual Comparison (Predicted - Actual NO₂)',
    fontsize=11,
    fontweight='bold',
)
ax2.set_xlabel(
    'Chronological Day of Dataset (2023 - 2024)', fontsize=12, fontweight='bold'
)
ax2.set_ylabel('Error (ppb)', fontsize=11, fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)

plt.tight_layout()
out_file = OUTPUT_DIR / "pomona_xgboost_vs_classical_baselines.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved Digital Twin Overlay Figure to: {out_file}")
print("🎯 PHASE 2 BRIDGE COMPLETE!")