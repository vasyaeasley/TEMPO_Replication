import os
from pathlib import Path
import time
import zipfile
import netCDF4 as nc
import numpy as np
from scipy.spatial import cKDTree

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOWNLOAD_DIR = BASE_DIR / "data" / "raw" / "era5_monthly_chunks"

archive_18_path = PROCESSED_DIR / "epa_point_dataset_14months_18features.npz"
final_20_path = PROCESSED_DIR / "epa_point_dataset_14months_20features.npz"

print("🌧️ STARTING NATIVE-GRID WEATHER INJECTION: 18 -> 20 COVARIATES 🚀")
print("=" * 75)

if not archive_18_path.exists():
  raise FileNotFoundError(
      f"Could not find the 18-feature baseline archive at {archive_18_path}"
  )


# ==============================================================================
# 2. DIAGNOSTIC FILE INSPECTOR & AUTO-UNZIPPER
# ==============================================================================
def safe_open_netcdf(file_path):
  if not file_path.exists():
    return None

  file_size = file_path.stat().st_size
  if file_size < 500:
    return None

  try:
    with open(file_path, "rb") as f:
      magic = f.read(4)

    if magic.startswith(b"PK"):
      with zipfile.ZipFile(file_path, "r") as z:
        target_names = [
            name
            for name in z.namelist()
            if name.endswith(".nc") or name.endswith(".grib")
        ]
        if not target_names:
          target_names = z.namelist()

        extracted_path = DOWNLOAD_DIR / f"{file_path.stem}_unzipped.nc"
        with open(extracted_path, "wb") as f_out:
          f_out.write(z.read(target_names[0]))

      return nc.Dataset(extracted_path, "r")
  except Exception:
    pass

  try:
    return nc.Dataset(file_path, "r")
  except OSError:
    return None


# ==============================================================================
# 3. LOAD THE 18-FEATURE ARRAYS & STATION COORDINATES
# ==============================================================================
print("⏳ Loading 18-feature state-wide matrices...")
base_data = np.load(archive_18_path, allow_pickle=True)

X_train = base_data["X_train"]
y_train = base_data["y_train"]
X_test = base_data["X_test"]
y_test = base_data["y_test"]
g_train = base_data["groups_train"]
g_test = base_data["groups_test"]
base_features = list(base_data["feature_names"])

print("🌐 Reconstructing station GPS coordinates...")
master_file = PROCESSED_DIR / "era5_california_1x1km_master.nc"
airnow_dir = BASE_DIR / "data" / "raw" / "epa" / "AirNow"
if not airnow_dir.exists():
  airnow_dir = BASE_DIR / "data" / "raw" / "AirNow"

with nc.Dataset(master_file, "r") as m_ds:
  lat_key = "lat" if "lat" in m_ds.variables else "latitude"
  lon_key = "lon" if "lon" in m_ds.variables else "longitude"
  grid_lats = m_ds.variables[lat_key][:]
  grid_lons = m_ds.variables[lon_key][:]

airnow_files = sorted(list(airnow_dir.glob("*.nc")))
with nc.Dataset(airnow_files[0], "r") as a_ds:
  stn_lats = a_ds.variables["latitude"][:]
  stn_lons = a_ds.variables["longitude"][:]

valid_mask = (
    (stn_lats >= np.min(grid_lats))
    & (stn_lats <= np.max(grid_lats))
    & (stn_lons >= np.min(grid_lons))
    & (stn_lons <= np.max(grid_lons))
)
valid_indices = np.where(valid_mask)[0]
ca_lats = stn_lats[valid_indices]
ca_lons = stn_lons[valid_indices]

# ==============================================================================
# 4. INTERPOLATE D2M AND TCC USING NATIVE 43x45 GRID SNAPPING
# ==============================================================================
print(
    "\n⏳ Mapping Dewpoint (d2m) and Cloud Cover (tcc) from native 43x45"
    " grids..."
)


def process_matrix(X_mat, groups, name="Dataset"):
  d2m_new = np.zeros(len(X_mat))
  tcc_new = np.zeros(len(X_mat))

  hour_sin, hour_cos = X_mat[:, 8], X_mat[:, 9]
  month_sin, month_cos = X_mat[:, 12], X_mat[:, 13]

  hours = np.round(np.arctan2(hour_sin, hour_cos) * 24.0 / (2 * np.pi)) % 24
  months = np.round(np.arctan2(month_sin, month_cos) * 12.0 / (2 * np.pi)) % 12
  months = np.where(months == 0, 12, months)

  unique_months = np.unique(months)

  for m in unique_months:
    m_mask = months == m
    year = 2024 if m < 8 else 2023

    unzipped_file = (
        DOWNLOAD_DIR
        / f"era5_california_missing_vars_{year}_{int(m):02d}_unzipped.nc"
    )
    file_path = (
        DOWNLOAD_DIR / f"era5_california_missing_vars_{year}_{int(m):02d}.nc"
    )

    if not file_path.exists() and not unzipped_file.exists():
      file_path = (
          DOWNLOAD_DIR / f"era5_california_missing_vars_2024_{int(m):02d}.nc"
      )

    target_path = unzipped_file if unzipped_file.exists() else file_path
    print(f"   -> Processing Month {year}-{int(m):02d} ({name})...")

    ds = safe_open_netcdf(target_path)
    if ds is None:
      continue

    try:
      t_key = "time" if "time" in ds.variables else "valid_time"
      dates = nc.num2date(ds.variables[t_key][:], ds.variables[t_key].units)

      # Dynamically build spatial lookup for this exact Copernicus file's grid (e.g., 43x45)
      f_lat_key = (
          "latitude"
          if "latitude" in ds.variables
          else ("lat" if "lat" in ds.variables else None)
      )
      f_lon_key = (
          "longitude"
          if "longitude" in ds.variables
          else ("lon" if "lon" in ds.variables else None)
      )
      f_lats = ds.variables[f_lat_key][:]
      f_lons = ds.variables[f_lon_key][:]

      if f_lats.ndim == 1:
        f_lon_grid, f_lat_grid = np.meshgrid(f_lons, f_lats)
      else:
        f_lat_grid, f_lon_grid = f_lats, f_lons

      f_width = f_lat_grid.shape[1]
      f_tree = cKDTree(
          np.column_stack([f_lat_grid.ravel(), f_lon_grid.ravel()])
      )

      # Query station pixel locations specifically against this 43x45 raster
      _, f_pixel_indices = f_tree.query(np.column_stack([ca_lats, ca_lons]))
      native_rows = f_pixel_indices // f_width
      native_cols = f_pixel_indices % f_width

      # Extract variable arrays
      d2m_raw = np.ma.filled(ds.variables["d2m"][:].astype("float64"), np.nan)
      tcc_raw = np.ma.filled(ds.variables["tcc"][:].astype("float64"), np.nan)

      idx_list = np.where(m_mask)[0]
      for idx in idx_list:
        g_id = int(groups[idx])
        r, c = native_rows[g_id], native_cols[g_id]
        h = int(hours[idx])
        t_idx = min(h, len(dates) - 1)

        d2m_new[idx] = (np.nan_to_num(d2m_raw[t_idx, r, c]) - 273.15) / 10.0
        tcc_new[idx] = np.nan_to_num(tcc_raw[t_idx, r, c])
    except Exception as e:
      print(f"      ⚠️ Error reading variables for Month {m}: {e}")
    finally:
      ds.close()

  return np.column_stack([d2m_new, tcc_new])


print("⚙️ Processing Training matrix...")
X_train_new = process_matrix(X_train, g_train, "Train")
X_train_final = np.column_stack([X_train, X_train_new])

print("⚙️ Processing Testing matrix...")
X_test_new = process_matrix(X_test, g_test, "Test")
X_test_final = np.column_stack([X_test, X_test_new])

# ==============================================================================
# 5. SAVE COMPRESSED 20-FEATURE ARCHIVE
# ==============================================================================
new_features = ["d2m", "tcc"]
final_feature_names = np.array(base_features + new_features)

np.savez_compressed(
    final_20_path,
    X_train=X_train_final,
    y_train=y_train,
    X_test=X_test_final,
    y_test=y_test,
    groups_train=g_train,
    groups_test=g_test,
    feature_names=final_feature_names,
)

print("\n" + "=" * 75)
print("🎉 MASTER 20-FEATURE DATASET COMPILED SUCCESSFULLY!")
print(f"💾 File Saved to: {final_20_path}")
print(f"   * Expanded Matrix Dimensions: {X_train_final.shape}")
print("=" * 75)