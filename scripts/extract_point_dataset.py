from datetime import datetime, timedelta
from pathlib import Path
import time
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. SETUP & CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
processed_dir = BASE_DIR / 'data' / 'processed'
output_dir = BASE_DIR / 'data' / 'processed'
model_dir = BASE_DIR / 'models'
model_dir.mkdir(exist_ok=True)

master_file = processed_dir / 'era5_california_1x1km_master.nc'
pop_file = processed_dir / 'globpop_1x1km.nc'
elev_file = processed_dir / 'usgs_elevation_1x1km.nc'
traffic_file = processed_dir / 'traffic_aadt_1x1km.nc'
road_file = processed_dir / 'grip_road_density_1x1km.nc'
tempo_dir = processed_dir / 'tempo_monthly'

airnow_dir = BASE_DIR / 'data' / 'raw' / 'epa' / 'AirNow'
if not airnow_dir.exists():
  airnow_dir = BASE_DIR / 'data' / 'raw' / 'AirNow'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print('🌍 SPATIALLY-AWARE 14-MONTH STATE-WIDE POINT EXTRACTION PIPELINE 🌍')
print('=' * 75)

# ==============================================================================
# 2. BUILD KD-TREE FOR INSTANT GPS -> GRID PIXEL SNAPPING
# ==============================================================================
print('⏳ Loading Master Grid Coordinates & Building KD-Tree...')
with nc.Dataset(master_file, 'r') as m_ds:
  lat_key = 'lat' if 'lat' in m_ds.variables else 'latitude'
  lon_key = 'lon' if 'lon' in m_ds.variables else 'longitude'
  grid_lats = m_ds.variables[lat_key][:]
  grid_lons = m_ds.variables[lon_key][:]

  if grid_lats.ndim == 1:
    lon_grid, lat_grid = np.meshgrid(grid_lons, grid_lats)
  else:
    lat_grid, lon_grid = grid_lats, grid_lons

  height, width = lat_grid.shape

grid_coords = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
kdtree = cKDTree(grid_coords)
print(
    f'✅ KD-Tree built across {height * width:,} pixels! ({height} rows * {width}'
    ' cols)'
)

# ==============================================================================
# 3. EXTRACT & CLUSTER EPA SENSOR LOCATIONS (K-Means k=5 across ALL CA!)
# ==============================================================================
print('\n⏳ Scanning AirNow monitoring stations...')
airnow_files = sorted(list(airnow_dir.glob('*.nc')))
if not airnow_files:
  raise FileNotFoundError('No AirNow files found!')

with nc.Dataset(airnow_files[0], 'r') as a_ds:
  stn_lats = a_ds.variables['latitude'][:]
  stn_lons = a_ds.variables['longitude'][:]
  stn_ids = [str(sid).strip() for sid in a_ds.variables['site'][:]]

# --- FULL CALIFORNIA STATE-WIDE BOUNDING BOX ---
# Dynamically matches all stations inside our master California grid boundaries!
lat_min, lat_max = np.min(grid_lats), np.max(grid_lats)
lon_min, lon_max = np.min(grid_lons), np.max(grid_lons)

print(
    f'\n🌍 Applying Full California Grid Bounds: Lat [{lat_min:.2f},'
    f' {lat_max:.2f}] | Lon [{lon_min:.2f}, {lon_max:.2f}]'
)

valid_mask = (
    (stn_lats >= lat_min)
    & (stn_lats <= lat_max)
    & (stn_lons >= lon_min)
    & (stn_lons <= lon_max)
)
valid_indices = np.where(valid_mask)[0]
# -----------------------------------------------

ca_lats = stn_lats[valid_indices]
ca_lons = stn_lons[valid_indices]
ca_ids = [stn_ids[i] for i in valid_indices]
n_stations = len(ca_lats)

print(
    f'✅ Found {n_stations} valid EPA monitoring stations across all of'
    ' California!'
)

_, pixel_indices = kdtree.query(np.column_stack([ca_lats, ca_lons]))
stn_rows = pixel_indices // width
stn_cols = pixel_indices % width

print('\n🎯 Running K-Means (k=5) Spatially-Aware State-Wide Sensor Split...')
kmeans = KMeans(n_clusters=5, random_state=RANDOM_SEED, n_init=10)
clusters = kmeans.fit_predict(np.column_stack([ca_lats, ca_lons]))

train_set = set()
test_set = set()

for c_id in range(5):
  c_mask = np.where(clusters == c_id)[0]
  c_train, c_test = train_test_split(
      c_mask, test_size=0.40, random_state=RANDOM_SEED
  )
  train_set.update(c_train)
  test_set.update(c_test)

print(
    f'✅ Split Complete -> Training Stations: {len(train_set)} (60%) | Testing'
    f' Stations: {len(test_set)} (40%)'
)

# ==============================================================================
# 4. PRE-LOAD STATIC LAYERS & BUILD MASTER STATION CATALOG
# ==============================================================================
print('\n⏳ Pre-loading Static Features at Sensor Locations...')

with nc.Dataset(pop_file, 'r') as p_ds:
  pop_var = next(
      v
      for v in p_ds.variables.keys()
      if v.lower()
      not in [
          'lat',
          'lon',
          'latitude',
          'longitude',
          'time',
          'crs',
          'band',
          'spatial_ref',
      ]
  )
  pop_raw = np.asarray(p_ds.variables[pop_var][:], dtype=np.float64)
  pop_raw = np.where(pop_raw < -1e5, np.nan, pop_raw)
  pop_clean = np.clip(np.nan_to_num(pop_raw, nan=0.0), a_min=0.0, a_max=None)
  pop_static = np.log1p(pop_clean) / 10.0

with nc.Dataset(elev_file, 'r') as e_ds:
  elev_var = next(
      v
      for v in e_ds.variables.keys()
      if v.lower()
      not in [
          'lat',
          'lon',
          'latitude',
          'longitude',
          'time',
          'crs',
          'band',
          'spatial_ref',
      ]
  )
  elev_raw = np.asarray(e_ds.variables[elev_var][:], dtype=np.float64)
  elev_raw = np.where(elev_raw < -1e5, np.nan, elev_raw)
  elev_static = np.nan_to_num(elev_raw, nan=0.0) / 4000.0

with nc.Dataset(traffic_file, 'r') as tr_ds:
  traffic_static = np.nan_to_num(
      np.asarray(tr_ds.variables['traffic_aadt'][:], dtype=np.float64), nan=0.0
  )

with nc.Dataset(road_file, 'r') as rd_ds:
  road_static = np.nan_to_num(
      np.asarray(rd_ds.variables['road_density'][:], dtype=np.float64), nan=0.0
  )

master_stations = {}
for i, sid in enumerate(ca_ids):
  master_stations[sid] = {
      'gid': i,
      'row': stn_rows[i],
      'col': stn_cols[i],
      'pop': pop_static[stn_rows[i], stn_cols[i]],
      'elev': elev_static[stn_rows[i], stn_cols[i]],
      'is_train': (i in train_set),
  }

print(
    f'✅ Master Station Catalog built for all {len(master_stations)} California'
    ' sites!'
)

# ==============================================================================
# 5. THE 14-MONTH DYNAMIC TIME-SERIES EXTRACTION LOOP (WITH 18 FEATURES!)
# ==============================================================================
print(
    '\n⏳ Extracting 18 Hourly Covariates across 14 Months of Daily Files...'
)
start_time = time.time()

X_train_list, y_train_list = [], []
X_test_list, y_test_list = [], []
g_train_list, g_test_list = [], []

era5_ds = nc.Dataset(master_file, 'r')
time_key = (
    'time'
    if 'time' in era5_ds.variables
    else (
        'valid_time'
        if 'valid_time' in era5_ds.variables
        else list(era5_ds.dimensions.keys())[0]
    )
)

var_map = {}
for k, cands in {
    't2m': ['t2m', '2m_temperature', 'temp'],
    'u10': ['u10', '10m_u_wind', 'u_wind'],
    'v10': ['v10', '10m_v_wind', 'v_wind'],
    'sp': ['sp', 'surface_pressure', 'pressure'],
    'blh': ['blh', 'boundary_layer_height', 'pblh'],
}.items():
  for c in cands:
    if c in era5_ds.variables:
      var_map[c] = c
      var_map[k] = c
      break

tempo_files = sorted(list(tempo_dir.glob('*.nc')))


def _parse_since(units):
  _, _, ref = units.partition(' since ')
  ref = ref.strip().replace('T', ' ')
  date_part, _, time_part = ref.partition(' ')
  time_part = time_part or '00:00:00'
  if '.' in time_part:
    hms, frac = time_part.split('.')
    time_part = f"{hms}.{(frac + '000000')[:6]}"
    fmt = '%Y-%m-%d %H:%M:%S.%f'
  else:
    fmt = '%Y-%m-%d %H:%M:%S'
  return datetime.strptime(f'{date_part} {time_part}', fmt)


def _round_hour(d):
  return (d + timedelta(minutes=30)).replace(minute=0, second=0, microsecond=0)


pop_arr = pop_static[stn_rows, stn_cols]
elev_arr = elev_static[stn_rows, stn_cols]
traffic_arr = traffic_static[stn_rows, stn_cols]
road_arr = road_static[stn_rows, stn_cols] / 10000.0
is_train_arr = np.array([(i in train_set) for i in range(n_stations)])

era5_dates = nc.num2date(
    era5_ds.variables[time_key][:], era5_ds.variables[time_key].units
)
era5_idx = {
    (d.year, d.month, d.day, d.hour): i for i, d in enumerate(era5_dates)
}
MET_VARS = ['t2m', 'u10', 'v10', 'sp', 'blh']
era5_cache = {}


def era5_at_stations(ei):
  cached = era5_cache.get(ei)
  if cached is None:
    cached = {}
    for vk in MET_VARS:
      grid = np.ma.filled(
          era5_ds.variables[var_map[vk]][ei, :, :].astype('float64'), np.nan
      )
      cached[vk] = grid[stn_rows, stn_cols]
    era5_cache[ei] = cached
  return cached


airnow_by_date = {}
for f in airnow_files:
  with nc.Dataset(f, 'r') as a_ds:
    ids = [str(s).strip() for s in a_ds.variables['site'][:]]
    no2 = np.ma.filled(a_ds.variables['no2'][:].astype('float64'), np.nan)
  airnow_by_date[f.stem.split('_')[-1]] = (ids, no2)

total_samples_extracted = 0

for m_idx, t_file in enumerate(tempo_files):
  m_name = t_file.stem.replace('regridded_', '').replace('tempo_', '')
  print(
      f'   [{m_idx + 1:2d}/{len(tempo_files)}] Month {m_name}...',
      end='',
      flush=True,
  )

  with nc.Dataset(t_file, 'r') as t_ds:
    no2_tempo = np.ma.filled(
        t_ds.variables['NO2_column'][:].astype('float64'), np.nan
    )
    t_raw = np.asarray(t_ds.variables['time'][:])
    t_base = _parse_since(t_ds.variables['time'].units)

  tempo_at_stn = no2_tempo[:, stn_rows, stn_cols]
  month_samples = 0

  for k in range(len(t_raw)):
    stamp = _round_hour(
        t_base + timedelta(microseconds=float(t_raw[k]) / 1000.0)
    )
    ei = era5_idx.get((stamp.year, stamp.month, stamp.day, stamp.hour))
    if ei is None:
      continue
    rec = airnow_by_date.get(
        f'{stamp.year:04d}{stamp.month:02d}{stamp.day:02d}'
    )
    if rec is None:
      continue
    daily_ids, daily_no2 = rec

    active_indices, active_gids = [], []
    for idx, sid in enumerate(daily_ids):
      info = master_stations.get(sid)
      if info is not None:
        active_indices.append(idx)
        active_gids.append(info['gid'])
    if not active_indices:
      continue
    active_indices = np.array(active_indices)
    active_gids = np.array(active_gids)

    met = era5_at_stations(ei)
    hour = stamp.hour
    y_vals = daily_no2[hour, active_indices]
    t_vals = tempo_at_stn[k, active_gids] / 1e15
    t2m_vals = (met['t2m'][active_gids] - 285.0) / 15.0
    u10_vals = met['u10'][active_gids] / 10.0
    v10_vals = met['v10'][active_gids] / 10.0
    sp_vals = (met['sp'][active_gids] - 95000.0) / 5000.0
    blh_vals = met['blh'][active_gids] / 1000.0

    # 1-2. Diurnal Clock
    hour_sin_vals = np.full(len(active_gids), np.sin(2.0 * np.pi * hour / 24.0))
    hour_cos_vals = np.full(len(active_gids), np.cos(2.0 * np.pi * hour / 24.0))

    # =========================================================================
    # 🚀 NEW: 6 FREE SEASONAL & ASTRONOMICAL COVARIATES (PATH A UPGRADE)
    # =========================================================================
    # 3-4. Annual Seasonality (Month sin/cos) - Cures winter inversion blindness!
    month = stamp.month
    month_sin_vals = np.full(
        len(active_gids), np.sin(2.0 * np.pi * month / 12.0)
    )
    month_cos_vals = np.full(
        len(active_gids), np.cos(2.0 * np.pi * month / 12.0)
    )

    # 5-6. Weekly Cycles (Day of Week sin/cos)
    dow = stamp.weekday()
    dow_sin_vals = np.full(len(active_gids), np.sin(2.0 * np.pi * dow / 7.0))
    dow_cos_vals = np.full(len(active_gids), np.cos(2.0 * np.pi * dow / 7.0))

    # 7. Weekend Flag (Fixes Saturday/Sunday industrial freight overprediction)
    is_weekend_vals = np.full(len(active_gids), 1.0 if dow >= 5 else 0.0)

    # 8. Solar Zenith Angle (SZA) - Direct Astronomical Proxy for UV Photolysis!
    day_of_year = stamp.timetuple().tm_yday
    lat_rad = np.radians(ca_lats[active_gids])

    gamma = 2 * np.pi / 365.0 * (day_of_year - 1 + (hour - 12) / 24.0)
    declination = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
    )
    hour_angle = np.radians((hour - 12.0) * 15.0)

    cos_sza = np.sin(lat_rad) * np.sin(declination) + np.cos(lat_rad) * np.cos(
        declination
    ) * np.cos(hour_angle)
    sza_vals = (
        np.degrees(np.arccos(np.clip(cos_sza, -1.0, 1.0))) / 90.0
    )  # Normalized ~0..1
    # =========================================================================

    X_matrix = np.column_stack([
        t_vals,
        t2m_vals,
        u10_vals,
        v10_vals,
        sp_vals,
        blh_vals,
        pop_arr[active_gids],
        elev_arr[active_gids],
        hour_sin_vals,
        hour_cos_vals,
        traffic_arr[active_gids],
        road_arr[active_gids],
        # Our 6 new upgrades:
        month_sin_vals,
        month_cos_vals,
        dow_sin_vals,
        dow_cos_vals,
        is_weekend_vals,
        sza_vals,
    ])

    valid_mask = (
        ~np.isnan(y_vals)
        & (y_vals > 0)
        & ~np.isnan(t_vals)
        & (t_vals > 0)
        & ~np.isnan(t2m_vals)
    )

    active_is_train = is_train_arr[active_gids]
    train_valid = valid_mask & active_is_train
    if np.any(train_valid):
      X_train_list.append(X_matrix[train_valid])
      y_train_list.append(y_vals[train_valid])
      g_train_list.append(active_gids[train_valid])
      month_samples += int(np.sum(train_valid))

    test_valid = valid_mask & ~active_is_train
    if np.any(test_valid):
      X_test_list.append(X_matrix[test_valid])
      y_test_list.append(y_vals[test_valid])
      g_test_list.append(active_gids[test_valid])
      month_samples += int(np.sum(test_valid))

  total_samples_extracted += month_samples
  print(f' matched {month_samples:7,d} station-hour pairs.')

era5_ds.close()

if not X_train_list or not X_test_list:
  raise RuntimeError('No data pairs were extracted!')

X_train = np.vstack(X_train_list)
y_train = np.concatenate(y_train_list)
X_test = np.vstack(X_test_list)
y_test = np.concatenate(y_test_list)
g_train = np.concatenate(g_train_list)
g_test = np.concatenate(g_test_list)

duration = time.time() - start_time
print('=' * 75)
print(f'🎉 STATE-WIDE CALIFORNIA EXTRACTION FINISHED IN {duration:.1f} SECONDS!')
print('=' * 75)
print(
    '   Train Dataset Shape (60% Unseen CA Sensors):'
    f' X={X_train.shape} | y={y_train.shape}'
)
print(
    '   Test Dataset Shape  (40% Unseen CA Sensors):'
    f' X={X_test.shape}  | y={y_test.shape}'
)
print('=' * 75)

# Save as our upgraded 18-feature state-wide archive!
save_path = output_dir / 'epa_point_dataset_14months_18features.npz'
feature_names = np.array([
    'TEMPO_NO2',
    't2m',
    'u10',
    'v10',
    'sp',
    'blh',
    'pop',
    'elev',
    'hour_sin',
    'hour_cos',
    'traffic',
    'road_density',
    'month_sin',
    'month_cos',
    'day_of_week_sin',
    'day_of_week_cos',
    'is_weekend',
    'solar_zenith_angle',
])
np.savez_compressed(
    save_path,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    groups_train=g_train,
    groups_test=g_test,
    feature_names=feature_names,
)
print(f'💾 Compressed 18-Feature California dataset saved to:\n   {save_path}')
print('\n🎯 YOU ARE READY TO TRAIN THE UPGRADED 18-FEATURE STATE-WIDE XGBOOST!')