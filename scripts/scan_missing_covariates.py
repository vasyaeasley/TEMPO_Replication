import os
from pathlib import Path
import numpy as np
import pandas as pd

# Safely import NetCDF/HDF5 inspection libraries
try:
  import netCDF4 as nc

  HAS_NC = True
except ImportError:
  HAS_NC = False

try:
  import xarray as xr

  HAS_XR = True
except ImportError:
  HAS_XR = False

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
if not RAW_DIR.exists():
  RAW_DIR = BASE_DIR / "raw"

print("🔍 STARTING DIAGNOSTIC SCAN: HUNTING FOR MISSING COVARIATES 🔍")
print("=" * 75)

# Target variables from Kenchammana et al. (2024) and physical air quality literature
target_era5 = {
    "ssrd": "Surface Solar Radiation Downwards (UV / Photolysis driver)",
    "d2m": "2-meter Dewpoint Temperature (Relative Humidity / Aerosol driver)",
    "tcc": "Total Cloud Cover (Radiation / Photolysis blocker)",
    "tp": "Total Precipitation (Atmospheric rain scrubber)",
}

target_tempo = {
    "solar_zenith_angle": "Solar Zenith Angle (SZA - Optical path length)",
    "viewing_zenith_angle": "Viewing Zenith Angle (VZA - Sensor look angle)",
    "cloud_fraction": "Cloud Fraction (Sub-pixel smog masking)",
    "amf_cloud_fraction": "Air Mass Factor Cloud Fraction",
    "cloud_pressure": "Cloud Top Pressure",
}

found_era5 = set()
found_tempo = set()

# ==============================================================================
# 2. SCANNING ECMWF ERA5 DIRECTORIES (.NC / .GRIB / .CSV)
# ==============================================================================
print("⏳ Scanning directories for ECMWF ERA5 weather files...")
era5_files = list(RAW_DIR.glob("**/*era5*")) + list(RAW_DIR.glob("**/*weather*"))
era5_files = [
    f
    for f in era5_files
    if f.suffix in [".nc", ".grib", ".grib2", ".csv", ".npz"]
]

if not era5_files:
  # Fallback: scan all nc files in raw
  era5_files = list(RAW_DIR.glob("**/*.nc"))

print(
    f"📂 Found {len(era5_files)} potential atmospheric weather/ERA5 files to"
    " inspect."
)

for f_path in era5_files[:5]:  # Inspect first 5 files to map dataset structure
  print(f"\n📄 Inspecting: {f_path.name}")
  if f_path.suffix == ".csv":
    df_sample = pd.read_csv(f_path, nrows=5)
    cols = df_sample.columns.tolist()
    print(f"   * CSV Columns: {cols}")
    for col in cols:
      for k in target_era5.keys():
        if k.lower() in col.lower():
          found_era5.add(k)
  elif f_path.suffix == ".nc" and HAS_NC:
    try:
      ds = nc.Dataset(f_path)
      vars_found = list(ds.variables.keys())
      print(f"   * NetCDF Variables: {vars_found}")
      for v in vars_found:
        if v.lower() in target_era5:
          found_era5.add(v.lower())
      ds.close()
    except Exception as e:
      print(f"   * Could not read NetCDF ({e})")
  elif f_path.suffix == ".nc" and HAS_XR and not HAS_NC:
    try:
      ds = xr.open_dataset(f_path)
      vars_found = list(ds.data_vars.keys())
      print(f"   * Xarray Variables: {vars_found}")
      for v in vars_found:
        if v.lower() in target_era5:
          found_era5.add(v.lower())
      ds.close()
    except Exception as e:
      print(f"   * Could not read Xarray ({e})")

# ==============================================================================
# 3. SCANNING NASA TEMPO DIRECTORIES (.NC / .H5 / .HE5)
# ==============================================================================
print("\n" + "=" * 75)
print("⏳ Scanning directories for NASA TEMPO / TROPOMI satellite files...")
tempo_files = (
    list(RAW_DIR.glob("**/*tempo*"))
    + list(RAW_DIR.glob("**/*tropomi*"))
    + list(RAW_DIR.glob("**/*NO2*"))
)
tempo_files = [
    f for f in tempo_files if f.suffix in [".nc", ".h5", ".he5", ".csv"]
]

print(f"📂 Found {len(tempo_files)} potential satellite files to inspect.")

for f_path in tempo_files[:5]:
  print(f"\n📄 Inspecting: {f_path.name}")
  if f_path.suffix == ".csv":
    df_sample = pd.read_csv(f_path, nrows=5)
    cols = df_sample.columns.tolist()
    print(f"   * CSV Columns: {cols}")
    for col in cols:
      for k in target_tempo.keys():
        if k.lower() in col.lower() or "zenith" in col.lower():
          found_tempo.add(col)
  elif f_path.suffix in [".nc", ".h5", ".he5"] and HAS_NC:
    try:
      ds = nc.Dataset(f_path)
      # Inspect root variables and nested groups (standard in NASA L2/L3 files)
      root_vars = list(ds.variables.keys())
      groups = list(ds.groups.keys())
      print(f"   * Root Variables: {root_vars}")
      if groups:
        print(f"   * Nested Groups : {groups}")
        for g in groups:
          g_vars = list(ds.groups[g].variables.keys())
          print(f"     -> Group '{g}' contains: {g_vars}")
          for v in g_vars:
            if any(
                term in v.lower() for term in ["zenith", "cloud", "angle", "sza"]
            ):
              found_tempo.add(f"{g}/{v}")
      for v in root_vars:
        if any(term in v.lower() for term in ["zenith", "cloud", "angle", "sza"]):
          found_tempo.add(v)
      ds.close()
    except Exception as e:
      print(f"   * Could not read satellite archive ({e})")

# ==============================================================================
# 4. FINAL MISSING COVARIATE SCORECARD
# ==============================================================================
print("\n" + "=" * 75)
print("📋 12 MISSING COVARIATES SCORECARD & RECOVERY PLAN 📋")
print("=" * 75)

print("\n1️⃣ CALENDAR & SEASONAL TIMESTAMPS (No download required - 100% Free!):")
print(
    "   ✅ month_sin       : READY (Can be derived mathematically from Date"
    " Local)"
)
print(
    "   ✅ month_cos       : READY (Can be derived mathematically from Date"
    " Local)"
)
print(
    "   ✅ day_of_week_sin : READY (Can be derived mathematically from Date"
    " Local)"
)
print(
    "   ✅ day_of_week_cos : READY (Can be derived mathematically from Date"
    " Local)"
)
print(
    "   ✅ is_weekend      : READY (Captures weekend industrial freight drop)"
)

print("\n2️⃣ ECMWF ERA5 ATMOSPHERIC CHEMISTRY DRIVERS:")
for k, desc in target_era5.items():
  status = (
      "✅ DETECTED IN RAW FILES" if k in found_era5 else "⚠️ MISSING / NOT FOUND"
  )
  print(f"   * {k:6s} ({desc:55s}) -> {status}")

print("\n3️⃣ NASA TEMPO VIEWING GEOMETRY & QUALITY:")
if found_tempo:
  print("   ✅ DETECTED SATELLITE GEOMETRY/CLOUD VARIABLES IN ARCHIVES:")
  for v in found_tempo:
    print(f"      * Found: {v}")
else:
  print("   ⚠️ No viewing geometry or cloud variables explicitly auto-detected.")
  print(
      "      Note: If using L3 gridded files, geometry is often averaged out."
  )
  print(
      "      We can compute Solar Zenith Angle (SZA) astronomically using GPS"
      " + Timestamp!"
  )

print("\n" + "=" * 75)
print("🎯 DIAGNOSTIC SCAN COMPLETE!")