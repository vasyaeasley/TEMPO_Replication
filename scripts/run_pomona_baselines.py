import os

# Thread safety controls
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from datetime import datetime, timedelta
from pathlib import Path
import time
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

print('📍 STARTING LOCALIZED BASELINE REGRESSION SUITE: POMONA STATION 📍')
print('=' * 75)

if not data_file.exists():
  raise FileNotFoundError(f'Could not locate 20-feature dataset at: {data_file}')

# ==============================================================================
# 2. LOCATE POMONA STATION IN THE MASTER CATALOG
# ==============================================================================
print('⏳ Scanning AirNow metadata to isolate Pomona monitoring coordinates...')
airnow_files = sorted(list(airnow_dir.glob('*.nc')))

with nc.Dataset(airnow_files[0], 'r') as a_ds:
  stn_lats = a_ds.variables['latitude'][:]
  stn_lons = a_ds.variables['longitude'][:]
  stn_ids = [str(sid).strip() for sid in a_ds.variables['site'][:]]

# Pomona EPA AQS site is located near Lat 34.056, Lon -117.752
# Let's find the closest station in our valid California catalog!
pomona_lat, pomona_lon = 34.056, -117.752
stn_tree = cKDTree(np.column_stack([stn_lats, stn_lons]))
dist, idx = stn_tree.query([pomona_lat, pomona_lon])
pomona_id = stn_ids[idx]

print(
    f'🎯 Found Pomona Station! Site ID: {pomona_id} (Lat: {stn_lats[idx]:.3f},'
    f' Lon: {stn_lons[idx]:.3f})'
)

# Load 1x1km Master Grid to find Pomona's exact group ID (gid)
with nc.Dataset(master_file, 'r') as m_ds:
  lat_key = 'lat' if 'lat' in m_ds.variables else 'latitude'
  lon_key = 'lon' if 'lon' in m_ds.variables else 'longitude'
  grid_lats = m_ds.variables[lat_key][:]
  grid_lons = m_ds.variables[lon_key][:]
  if grid_lats.ndim == 1:
    lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
  else:
    lat_grid, lon_grid = grid_lats, grid_lons
  width = lat_grid.shape[1]

valid_mask = (
    (stn_lats >= np.min(grid_lats))
    & (stn_lats <= np.max(grid_lats))
    & (stn_lons >= np.min(grid_lons))
    & (stn_lons <= np.max(grid_lons))
)
valid_indices = np.where(valid_mask)[0]
ca_ids = [stn_ids[i] for i in valid_indices]

pomona_gid = ca_ids.index(pomona_id)
print(f'✅ Pomona Group ID (gid) in master matrix: {pomona_gid}')

# ==============================================================================
# 3. EXTRACT CHRONOLOGICAL TIME SERIES & CONVERT TO DAILY MEANS
# ==============================================================================
print('⏳ Loading 20-feature dataset and extracting Pomona time series...')
data = np.load(data_file, allow_pickle=True)

# Combine train and test splits to reconstruct the continuous 14-month timeline
X_full = np.vstack([data['X_train'], data['X_test']])
y_full = np.concatenate([data['y_train'], data['y_test']])
g_full = np.concatenate([data['groups_train'], data['groups_test']])

pomona_mask = g_full == pomona_gid
X_pom = X_full[pomona_mask]
y_pom = y_full[pomona_mask]

# Feature indices from our 20-feature build:
# 0:TEMPO_NO2, 1:t2m, 2:u10, 3:v10, 4:sp, 12:month_sin, 13:month_cos, 18:d2m, 19:tcc
# Derive timestamps from month_sin/cos and chronological sequence
hours_per_day = 9  # Average valid daylight satellite hours per day
n_days = len(y_pom) // hours_per_day

print(
    f'📊 Aggregating {len(y_pom):,} hourly observations into {n_days} Daily Mean'
    ' records...'
)

daily_no2, daily_t2m, daily_sp, daily_d2m, daily_u10, daily_v10, daily_doy = (
    [],
    [],
    [],
    [],
    [],
    [],
    [],
)
for i in range(n_days):
  slice_idx = slice(i * hours_per_day, (i + 1) * hours_per_day)
  daily_no2.append(np.mean(y_pom[slice_idx]))
  # Exponentiate normalized features back to physical weather units for regression:
  daily_t2m.append(
      np.mean(X_pom[slice_idx, 1]) * 15.0 + 285.0 - 273.15
  )  # Celsius
  daily_sp.append(
      (np.mean(X_pom[slice_idx, 4]) * 5000.0 + 95000.0) / 100.0
  )  # Millibars (hPa)
  daily_d2m.append(np.mean(X_pom[slice_idx, 18]) * 10.0)  # Dewpoint Celsius
  daily_u10.append(np.mean(X_pom[slice_idx, 2]) * 10.0)  # m/s
  daily_v10.append(np.mean(X_pom[slice_idx, 3]) * 10.0)  # m/s
  # Estimate Day of Year (DOY) roughly from chronological sequence (~August 2023 start)
  daily_doy.append((213 + i) % 365 + 1)

daily_no2 = np.array(daily_no2)
daily_t2m = np.array(daily_t2m)
daily_sp = np.array(daily_sp)
daily_d2m = np.array(daily_d2m)
daily_doy = np.array(daily_doy)

# Derive Wind Speed (knots) and Relative Humidity (%) using physical laws:
wind_speed_ms = np.sqrt(np.array(daily_u10) ** 2 + np.array(daily_v10) ** 2)
daily_wind = wind_speed_ms * 1.94384  # Convert m/s to knots

# Magnus-Tetens formula for Relative Humidity (%)
e_actual = 6.11 * np.exp((17.67 * daily_d2m) / (daily_d2m + 243.5))
e_sat = 6.11 * np.exp((17.67 * daily_t2m) / (daily_t2m + 243.5))
daily_rh = np.clip(100.0 * (e_actual / e_sat), 5.0, 100.0)

# ==============================================================================
# 4. PHASE 1 & 3: BUILD ALL BASELINE & STEPWISE REGRESSION MODELS
# ==============================================================================
print('\n⚙️ Fitting Climatology, Persistence, and Stepwise Regression models...')
results = []

def evaluate_model(name, pred_type, y_true, y_pred):
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  mae = mean_absolute_error(y_true, y_pred)
  r2 = r2_score(y_true, y_pred)
  results.append({
      'Model': name,
      'Prediction Type': pred_type,
      'RMSE (ppb)': rmse,
      'MAE (ppb)': mae,
      'R²': r2,
  })
  return y_pred


# 1. Climatological Mean (Constant prediction of overall mean)
y_mean = np.full(n_days, np.mean(daily_no2))
evaluate_model('Mean Baseline', 'Constant Overall Mean', daily_no2, y_mean)

# 2. Persistence Model (Lag-1: Yesterday's Daily Mean)
y_persist = np.roll(daily_no2, shift=1)
y_persist[0] = daily_no2[0]
evaluate_model('Persistence', 'Previous Day (Lag-1)', daily_no2, y_persist)

# 3. Simple Cosine Model (1 Harmonic DOY)
cos_doy = np.cos(2.0 * np.pi * (daily_doy - 15.0) / 365.0).reshape(-1, 1)
lr_cos = LinearRegression().fit(cos_doy, daily_no2)
y_cos = evaluate_model(
    'Cosine', 'Seasonal (1 Harmonic)', daily_no2, lr_cos.predict(cos_doy)
)

# 4. Harmonic Regression Model (2 Harmonics DOY)
X_harm = np.column_stack([
    np.sin(2 * np.pi * daily_doy / 365.0),
    np.cos(2 * np.pi * daily_doy / 365.0),
    np.sin(4 * np.pi * daily_doy / 365.0),
    np.cos(4 * np.pi * daily_doy / 365.0),
])
lr_harm = LinearRegression().fit(X_harm, daily_no2)
y_harm = evaluate_model(
    'Harmonic', 'Seasonal (2 Harmonics)', daily_no2, lr_harm.predict(X_harm)
)

# 5. Stepwise Model 1: Wind Speed Alone
X_wind = daily_wind.reshape(-1, 1)
lr_wind = LinearRegression().fit(X_wind, daily_no2)
y_wind = evaluate_model(
    'Stepwise 1-Param',
    'Wind Speed',
    daily_no2,
    lr_wind.predict(X_wind),
)

# 6. Stepwise Model 2: Wind Speed + Relative Humidity
X_w_rh = np.column_stack([daily_wind, daily_rh])
lr_w_rh = LinearRegression().fit(X_w_rh, daily_no2)
y_w_rh = evaluate_model(
    'Stepwise 2-Param',
    'Wind + RH',
    daily_no2,
    lr_w_rh.predict(X_w_rh),
)

# 7. Stepwise Model 3: Wind + RH + Temperature
X_w_rh_t = np.column_stack([daily_wind, daily_rh, daily_t2m])
lr_w_rh_t = LinearRegression().fit(X_w_rh_t, daily_no2)
y_w_rh_t = evaluate_model(
    'Stepwise 3-Param',
    'Wind + RH + Temp',
    daily_no2,
    lr_w_rh_t.predict(X_w_rh_t),
)

# 8. Stepwise Model 4: Full Meteorological Suite (Wind + RH + Temp + Pressure)
X_met_full = np.column_stack([daily_wind, daily_rh, daily_t2m, daily_sp])
lr_met_full = LinearRegression().fit(X_met_full, daily_no2)
y_met_full = evaluate_model(
    'Stepwise 4-Param',
    'Full Meteorology Suite',
    daily_no2,
    lr_met_full.predict(X_met_full),
)

# ==============================================================================
# 5. PRINT MASTER COMPARISON TABLE (SLIDE 33 & 36 FORMAT)
# ==============================================================================
print('\n' + '=' * 75)
print(f'📊 POMONA STATION (#060371701) BASELINE MODEL COMPARISON TABLE')
print('=' * 75)
df_res = pd.DataFrame(results).sort_values(by='RMSE (ppb)', ascending=True)
print(df_res.to_string(index=False))
print('=' * 75)

# ==============================================================================
# 6. GENERATE CORRELATION MATRIX & TIME SERIES FIGURES
# ==============================================================================
print('\n🎨 Generating Pomona Correlation Matrix and Time Series figures...')

# Figure 1: Correlation Matrix (Slide 23)
df_corr = pd.DataFrame({
    'NO₂ (ppb)': daily_no2,
    'Temperature (°F)': daily_t2m * 1.8 + 32.0,
    'Pressure (mb)': daily_sp,
    'Rel. Humidity (%)': daily_rh,
    'Wind Speed (knots)': daily_wind,
}).corr()

fig_c, ax_c = plt.subplots(figsize=(8, 7))
cax = ax_c.matshow(df_corr.values, cmap='RdBu_r', vmin=-1.0, vmax=1.0)
fig_c.colorbar(cax, fraction=0.046, pad=0.04, label='Correlation Coefficient')
ax_c.set_xticks(range(len(df_corr.columns)))
ax_c.set_yticks(range(len(df_corr.columns)))
ax_c.set_xticklabels(df_corr.columns, rotation=45, ha='left', fontweight='bold')
ax_c.set_yticklabels(df_corr.columns, fontweight='bold')
for i in range(len(df_corr.columns)):
  for j in range(len(df_corr.columns)):
    ax_c.text(
        j,
        i,
        f'{df_corr.values[i, j]:.3f}',
        ha='center',
        va='center',
        color='white' if abs(df_corr.values[i, j]) > 0.5 else 'black',
        fontweight='bold',
    )
ax_c.set_title(
    'Correlation Matrix: NO₂ & Meteorology\nPomona Station 2023-2024',
    fontsize=14,
    fontweight='bold',
    pad=20,
)
out_corr = OUTPUT_DIR / 'pomona_met_correlation_matrix.png'
plt.savefig(out_corr, dpi=300, bbox_inches='tight')
plt.close(fig_c)

# Figure 2: Time Series & Residuals (Slides 3, 6, 35)
days_x = np.arange(1, n_days + 1)
fig_t, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 9), sharex=True, gridspec_kw={'height_ratios': [2.2, 1]}
)

ax1.plot(
    days_x,
    daily_no2,
    label='Observed Daily Mean NO₂',
    color='#1f77b4',
    linewidth=1.8,
    alpha=0.8,
)
ax1.plot(
    days_x,
    y_persist,
    label='Persistence Baseline (Lag-1)',
    color='#2ca02c',
    linestyle='--',
    linewidth=1.5,
    alpha=0.7,
)
ax1.plot(
    days_x,
    y_harm,
    label='Harmonic Regression (Seasonal)',
    color='#9467bd',
    linewidth=2.0,
    alpha=0.85,
)
ax1.plot(
    days_x,
    y_met_full,
    label='Stepwise 4-Param Meteorology',
    color='#ff7f0e',
    linewidth=2.0,
    alpha=0.9,
)

ax1.set_title(
    'NO₂ Daily Mean Time Series & Baseline Model Comparisons\nPomona Monitoring Site'
    ' (#060371701)',
    fontsize=14,
    fontweight='bold',
    pad=12,
)
ax1.set_ylabel('NO₂ Concentration (ppb)', fontsize=12, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.95)

# Residuals plot
ax2.axhline(0, color='black', linewidth=1.2, linestyle='-', alpha=0.7)
ax2.plot(
    days_x,
    y_met_full - daily_no2,
    label='4-Param Meteorology Residuals',
    color='#ff7f0e',
    linewidth=1.2,
    alpha=0.8,
)
ax2.plot(
    days_x,
    y_persist - daily_no2,
    label='Persistence Residuals',
    color='#2ca02c',
    linestyle='--',
    linewidth=1.0,
    alpha=0.6,
)
ax2.set_title(
    'Model Residuals over Time (Predicted - Actual)',
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
out_time = OUTPUT_DIR / "pomona_baseline_timeseries_and_residuals.png"
plt.savefig(out_time, dpi=300, bbox_inches="tight")
plt.close(fig_t)

print(f"✅ Saved Correlation Matrix to: {out_corr}")
print(f"✅ Saved Time Series & Residuals to: {out_time}")
print("🎯 POMONA BASELINE REGRESSION SUITE COMPLETE!")