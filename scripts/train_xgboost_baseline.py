import time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
era5_file = BASE_DIR / 'data' / 'processed' / 'era5_california_1x1km_master.nc'
tempo_dir = BASE_DIR / 'data' / 'processed' / 'tempo_monthly'
output_dir = BASE_DIR / 'data' / 'processed'
model_dir = BASE_DIR / 'models'
model_dir.mkdir(exist_ok=True)

# We target 2.1 Million clean pixels across 14 months (~150,000 per month)
TOTAL_TARGET_SAMPLES = 2_100_000
SAMPLES_PER_MONTH = TOTAL_TARGET_SAMPLES // 14
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print('🌲 STRATIFIED DIURNAL XGBOOST BASELINE PIPELINE 🌲')
print(f'   Target Dataset Size: {TOTAL_TARGET_SAMPLES:,} clean pixels')
print(f'   Monthly Target:      {SAMPLES_PER_MONTH:,} pixels/month')

# ==============================================================================
# 2. UNIVERSAL VARIABLE MAPPING & METADATA
# ==============================================================================
tempo_files = sorted(list(tempo_dir.glob('*.nc')))
if not era5_file.exists() or not tempo_files:
  raise FileNotFoundError(
      'Missing ERA5 master file or TEMPO monthly swaths in cache!'
  )

with nc.Dataset(era5_file, 'r') as ds:
  time_key = (
      'time'
      if 'time' in ds.variables
      else (
          'valid_time'
          if 'valid_time' in ds.variables
          else list(ds.dimensions.keys())[0]
      )
  )
  expected_vars = {
      't2m': ['t2m', '2m_temperature', 'temp'],
      'u10': ['u10', '10m_u_wind', 'u_wind'],
      'v10': ['v10', '10m_v_wind', 'v_wind'],
      'sp': ['sp', 'surface_pressure', 'pressure'],
      'blh': ['blh', 'boundary_layer_height', 'pblh'],
  }
  var_map = {}
  for target, candidates in expected_vars.items():
    for cand in candidates:
      if cand in ds.variables:
        var_map[target] = cand
        break

print(f'✅ Master Grid Verified. Variable Mapping: {var_map}')

# ==============================================================================
# 3. STRATIFIED DIURNAL DATA SAMPLING
# ==============================================================================
print('\n⏳ Extracting Stratified Samples across 14 Months...')
start_sample_time = time.time()

X_list = []
y_list = []

# Open master file once for reading
era5_ds = nc.Dataset(era5_file, 'r')

for m_idx, t_file in enumerate(tempo_files):
  m_name = t_file.stem.replace('regridded_', '').replace('tempo_', '')
  print(f'   [{m_idx + 1}/14] Sampling {m_name}...', end='', flush=True)

  with nc.Dataset(t_file, 'r') as t_ds:
    no2_monthly = t_ds.variables['NO2_column'][:]
    n_hours, height, width = no2_monthly.shape

    # Calculate global time offset in the 300GB master file (~730 hrs/month)
    global_start_idx = m_idx * 730
    global_end_idx = min(global_start_idx + n_hours, era5_ds.variables[time_key].shape[0])
    actual_hours = global_end_idx - global_start_idx

    if actual_hours <= 0:
      print(' ⚠️ Skipped (Timeline mismatch)')
      continue

    # Slice matching weather covariates from the master cube
    t2m = era5_ds.variables[var_map['t2m']][global_start_idx:global_end_idx, :, :]
    u10 = era5_ds.variables[var_map['u10']][global_start_idx:global_end_idx, :, :]
    v10 = era5_ds.variables[var_map['v10']][global_start_idx:global_end_idx, :, :]
    sp = era5_ds.variables[var_map['sp']][global_start_idx:global_end_idx, :, :]
    blh = era5_ds.variables[var_map['blh']][global_start_idx:global_end_idx, :, :]

    # Slice target NO2 to match available hours
    no2 = no2_monthly[:actual_hours, :, :]

    # Flatten arrays to tabular format [Samples, 1]
    y_flat = no2.flatten()
    t2m_flat = t2m.flatten()
    u10_flat = u10.flatten()
    v10_flat = v10.flatten()
    sp_flat = sp.flatten()
    blh_flat = blh.flatten()

    # Find valid daytime pixels (not NaN, not ocean, positive NO2)
    valid_mask = (
        ~np.isnan(y_flat)
        & (y_flat > 0)
        & ~np.isnan(t2m_flat)
        & ~np.isnan(blh_flat)
    )
    valid_indices = np.where(valid_mask)[0]

    # Randomly sample our monthly target from valid pixels
    n_available = len(valid_indices)
    n_take = min(SAMPLES_PER_MONTH, n_available)

    if n_take > 0:
      chosen_idx = np.random.choice(valid_indices, size=n_take, replace=False)

      # Stack covariates into [n_take, 5] matrix
      X_month = np.column_stack([
          t2m_flat[chosen_idx],
          u10_flat[chosen_idx],
          v10_flat[chosen_idx],
          sp_flat[chosen_idx],
          blh_flat[chosen_idx],
      ])
      y_month = y_flat[chosen_idx]

      X_list.append(X_month)
      y_list.append(y_month)
      print(f' Got {n_take:,} pixels.')
    else:
      print(' ⚠️ No valid daytime pixels found.')

era5_ds.close()

# Concatenate all months into our unified training matrix
X_matrix = np.vstack(X_list)
y_vector = np.concatenate(y_list)

sample_duration = time.time() - start_sample_time
print(
    f'\n✅ Sampling Complete in {sample_duration:.1f}s! Total Dataset Shape: {X_matrix.shape}'
)
print(f'   Memory Footprint: {(X_matrix.nbytes + y_vector.nbytes) / (1024**2):.1f} MB')

# ==============================================================================
# 4. TRAIN / TEST SPLIT (80% Train, 20% Unseen Test)
# ==============================================================================
print('\n✂️ Splitting into 80% Training and 20% Testing sets...')
X_train, X_test, y_train, y_test = train_test_split(
    X_matrix, y_vector, test_size=0.20, random_state=RANDOM_SEED
)
print(f'   Train Rows: {X_train.shape[0]:,} | Test Rows: {X_test.shape[0]:,}')

# ==============================================================================
# 5. TRAINING THE XGBOOST ENSEMBLE
# ==============================================================================
print(
    '\n🚀 Initializing XGBoost Regressor (100 Trees, Max Depth 8, All CPU Cores)...'
)
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,  # Uses 100% of available CPU cores for maximum speed!
    random_state=RANDOM_SEED,
    tree_method='hist',  # High-speed histogram binning algorithm!
)

print('⏳ Training Model... (This should take ~1 to 3 minutes!)')
train_start = time.time()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=20,  # Prints progress every 20 trees!
)
train_duration = time.time() - train_start
print(f'🎉 Training Finished in {train_duration:.1f} seconds!')

# ==============================================================================
# 6. EVALUATION & BENCHMARKING
# ==============================================================================
print('\n📊 Calculating Replication Accuracy Metrics...')
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print('=' * 50)
print('🏆 XGBOOST REPLICATION RESULTS (TEST SET) 🏆')
print('=' * 50)
print(f'   R² Score (Explained Variance): {r2:.4f}')
print(
    f'   Root Mean Squared Error (RMSE): {rmse:.2e} molecules/cm²'
)
print(f'   Mean Absolute Error (MAE):     {mae:.2e} molecules/cm²')
print('=' * 50)

# Save Model for future use
model_path = model_dir / 'xgboost_tempo_baseline_14months.pkl'
joblib.dump(model, model_path)
print(f'💾 Model successfully saved to:\n   {model_path}')

# ==============================================================================
# 7. GENERATE FEATURE IMPORTANCE CHART FOR MENTOR
# ==============================================================================
print('\n🎨 Generating Feature Importance Chart...')
feature_names = [
    '2m Temperature',
    '10m U-Wind',
    '10m V-Wind',
    'Surface Pressure',
    'Boundary Layer Ht',
]
importances = model.feature_importances_

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(
    feature_names,
    importances * 100,
    color='forestgreen',
    edgecolor='black',
    alpha=0.85,
)

ax.set_xlabel('Relative Importance (%)', fontsize=12, fontweight='bold')
ax.set_title(
    'XGBoost Atmospheric Covariate Importance (14-Month California Digital Twin)',
    fontsize=14,
    fontweight='bold',
    pad=15,
)
ax.grid(axis='x', linestyle='--', alpha=0.5)

# Add value labels to bars
for bar in bars:
  width = bar.get_width()
  ax.text(
      width + 0.5,
      bar.get_y() + bar.get_height() / 2,
      f'{width:.1f}%',
      va='center',
      ha='left',
      fontsize=11,
      fontweight='bold',
  )

ax.set_xlim(0, max(importances * 100) * 1.15)
plt.tight_layout()

chart_path = output_dir / 'xgboost_feature_importance.png'
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
print(f'📈 Feature Importance chart saved to:\n   {chart_path}')
print('\n🎯 STAGE 1 BASELINE REPLICATION COMPLETELY CONQUERED!')