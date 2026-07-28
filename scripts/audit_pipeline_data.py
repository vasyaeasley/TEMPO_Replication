import os
import sys

# EXPLICIT THREAD-SAFETY CONTROLS - MUST PRECEDE NUMERICAL IMPORTS
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import numpy as np
import pandas as pd

# Attempt optional netCDF4 / xarray imports for satellite & reanalysis inspection
try:
    import netCDF4 as nc
except ImportError:
    nc = None

try:
    import xarray as xr
except ImportError:
    xr = None

BASE_DIR = "/mnt/data3/veasl001/TEMPO_Replication"

def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def audit_epa_hourly():
    print_header("1. EPA HOURLY CONTINUITY AUDIT (raw/epa_from_internet_hourly)")
    epa_dir = os.path.join(BASE_DIR, "raw/epa_from_internet_hourly")
    csv_files = sorted(glob.glob(os.path.join(epa_dir, "hourly_42602_*.csv")))
    
    if not csv_files:
        print(f"[!] No hourly NO2 CSVs found in {epa_dir}")
        return

    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        print(f"\n---> Auditing CSV: {fname}")
        df = pd.read_csv(csv_path, nrows=5000) # Peek structure first
        
        # Determine station unique identifier columns
        if set(['State Code', 'County Code', 'Site Num']).issubset(df.columns):
            df_full = pd.read_csv(csv_path, usecols=['State Code', 'County Code', 'Site Num', 'Date GMT', 'Time GMT', 'Sample Measurement'])
            df_full['station_id'] = (
                df_full['State Code'].astype(str).str.zfill(2) +
                df_full['County Code'].astype(str).str.zfill(3) +
                df_full['Site Num'].astype(str).str.zfill(4)
            )
            df_full['datetime_gmt'] = pd.to_datetime(df_full['Date GMT'] + ' ' + df_full['Time GMT'])
        elif 'site_id' in df.columns or 'AQS_SITE_ID' in df.columns:
            id_col = 'site_id' if 'site_id' in df.columns else 'AQS_SITE_ID'
            time_col = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()][0]
            val_col = [c for c in df.columns if 'sample' in c.lower() or 'val' in c.lower() or 'no2' in c.lower()][0]
            df_full = pd.read_csv(csv_path, usecols=[id_col, time_col, val_col])
            df_full['station_id'] = df_full[id_col].astype(str)
            df_full['datetime_gmt'] = pd.to_datetime(df_full[time_col])
            df_full['Sample Measurement'] = df_full[val_col]
        else:
            print(f"    Available columns: {list(df.columns)}")
            print("    [!] Custom column names detected. Inspect manually.")
            continue

        unique_stations = df_full['station_id'].unique()
        total_records = len(df_full)
        print(f"    Total Records: {total_records:,}")
        print(f"    Unique Stations: {len(unique_stations)}")
        print(f"    Date Range: {df_full['datetime_gmt'].min()} to {df_full['datetime_gmt'].max()}")

        # Check explicit temporal continuity on the primary station
        sample_station = unique_stations[0]
        st_data = df_full[df_full['station_id'] == sample_station].sort_values('datetime_gmt')
        
        # Theoretical complete hourly date range
        full_idx = pd.date_range(start=st_data['datetime_gmt'].min(), end=st_data['datetime_gmt'].max(), freq='h')
        actual_hours = len(st_data)
        expected_hours = len(full_idx)
        missing_hours = expected_hours - actual_hours
        null_measurements = st_data['Sample Measurement'].isna().sum()

        print(f"\n    [Sample Station {sample_station} Diagnostic]:")
        print(f"      Expected Hourly Timesteps : {expected_hours:,}")
        print(f"      Actual Rows in File       : {actual_hours:,}")
        print(f"      Completely Omitted Hours  : {missing_hours:,} ({missing_hours/expected_hours:.2%})")
        print(f"      NaN Measurement Rows     : {null_measurements:,}")
        
        if missing_hours > 0:
            print("    [RESULT] Rows for missing hours are OMITTED from CSV (Explicit re-indexing required for temporal derivatives).")
        else:
            print("    [RESULT] Continuous grid preserved with NaNs representing missing hours.")

def audit_tempo_files():
    print_header("2. TEMPO SATELLITE DATA AUDIT (raw/tempo)")
    tempo_dir = os.path.join(BASE_DIR, "raw/tempo")
    nc_files = sorted(glob.glob(os.path.join(tempo_dir, "**/*.nc*"), recursive=True))
    
    if not nc_files:
        print(f"[!] No NetCDF/HDF5 files found in {tempo_dir}")
        return

    print(f"Total TEMPO Granules Found: {len(nc_files)}")
    sample_file = nc_files[0]
    print(f"Sample Granule: {os.path.basename(sample_file)}")

    if nc is not None:
        try:
            ds = nc.Dataset(sample_file, 'r')
            print("\n    NetCDF4 Global Attributes / Groups:")
            print(f"      Groups: {list(ds.groups.keys()) if ds.groups else 'Root level only'}")
            
            # Look inside main product group if present
            prod_grp = ds.groups.get('product') if 'product' in ds.groups else ds
            print("\n    Variables inside main group:")
            for var_name in list(prod_grp.variables.keys())[:10]:
                v = prod_grp.variables[var_name]
                print(f"      - {var_name}: shape={v.shape}, dtype={v.dtype}")
            ds.close()
        except Exception as e:
            print(f"    [!] Error parsing NetCDF with netCDF4: {e}")
    elif xr is not None:
        try:
            ds = xr.open_dataset(sample_file)
            print("\n    xarray Dataset Summary:")
            print(f"      Dimensions: {dict(ds.dims)}")
            print(f"      Data Variables: {list(ds.data_vars.keys())}")
            ds.close()
        except Exception as e:
            print(f"    [!] Error parsing NetCDF with xarray: {e}")
    else:
        print("    [!] Neither netCDF4 nor xarray installed in tempo_env. Cannot inspect binary .nc4 metadata.")

def audit_era5_chunks():
    print_header("3. ERA5 METEOROLOGICAL REANALYSIS AUDIT (raw/era5_monthly_chunks)")
    era5_dir = os.path.join(BASE_DIR, "raw/era5_monthly_chunks")
    era5_files = sorted(glob.glob(os.path.join(era5_dir, "**/*"), recursive=True))
    era5_files = [f for f in era5_files if os.path.isfile(f)]

    if not era5_files:
        print(f"[!] No ERA5 files found in {era5_dir}")
        return

    print(f"Total ERA5 Files Found: {len(era5_files)}")
    exts = set([os.path.splitext(f)[1] for f in era5_files])
    print(f"Detected File Formats: {exts}")
    
    sample_file = era5_files[0]
    print(f"Sample ERA5 Chunk: {os.path.basename(sample_file)}")
    
    if xr is not None and any(ext in sample_file for ext in ['.nc', '.grib', '.nc4']):
        try:
            ds = xr.open_dataset(sample_file)
            print("\n    xarray ERA5 Spatial Structure:")
            print(f"      Dimensions: {dict(ds.dims)}")
            print(f"      Coordinates: {list(ds.coords.keys())}")
            print(f"      Variables: {list(ds.data_vars.keys())}")
            ds.close()
        except Exception as e:
            print(f"    [!] Could not parse ERA5 spatial file: {e}")

def audit_master_npz():
    print_header("4. PREPROCESSED MASTER DATASET AUDIT (data/processed/*.npz)")
    npz_path = os.path.join(BASE_DIR, "data/processed/epa_point_dataset_14months_20features.npz")
    
    if not os.path.exists(npz_path):
        print(f"[!] Master NPZ dataset not found at {npz_path}")
        return

    dataset = np.load(npz_path, allow_pickle=True)
    print(f"NPZ Dictionary Keys: {list(dataset.files)}")
    
    print("\nArray Shapes & Data Types:")
    for key in dataset.files:
        arr = dataset[key]
        print(f"  Key: '{key:<20}' | Shape: {str(arr.shape):<20} | Type: {arr.dtype}")

    # Matrix dimensionality evaluation
    first_key = dataset.files[0]
    primary_shape = dataset[first_key].shape
    
    print("\n[RESULT]:")
    if len(primary_shape) == 2:
        print("  The dataset is currently a FLAT 2D MATRIX (samples, features).")
        print("  --> To perform temporal shifts (dC/dt) or Graph Neural Network message passing,")
        print("      we must group by station ID and timestamp before forming 3D tensors.")
    elif len(primary_shape) == 3:
        print("  The dataset is ALREADY A 3D MATRIX (stations, timesteps, features).")
        print("  --> Immediate vectorized temporal derivative operations and spatial adjacency matrix calculations can be executed.")

if __name__ == "__main__":
    print(f"Executing Data Diagnostics in Environment: {sys.executable}")
    audit_epa_hourly()
    audit_tempo_files()
    audit_era5_chunks()
    audit_master_npz()
    print("\n" + "=" * 80)
    print(" AUDIT COMPLETE")
    print("=" * 80)