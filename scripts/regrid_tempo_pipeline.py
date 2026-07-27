import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import gc

# 1. Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
tempo_dir = BASE_DIR / 'data' / 'raw' / 'tempo' / 'NO2_L3_V03'
output_dir = BASE_DIR / 'data' / 'processed' / 'tempo_monthly'
output_dir.mkdir(parents=True, exist_ok=True)

print("1. Initializing 1x1 km Master California Grid...")
# California Bounding Box: [North: 42.5, West: -124.5, South: 32.5, East: -114.0]
# ~0.009 degrees roughly equals 1 km
lat_master = np.arange(32.5, 42.5, 0.009)
lon_master = np.arange(-124.5, -114.0, 0.009)

def load_and_regrid_single_file(filepath):
    """
    Opens root coordinates and /product/ data groups, applies NASA quality flags,
    and regrids onto the California 1x1 km master grid.
    """
    try:
        # Load coordinates from Root
        with xr.open_dataset(filepath) as ds_root:
            lats = ds_root['latitude'].values
            lons = ds_root['longitude'].values
            times = ds_root['time'].values

        # Load NO2 and quality flag from /product/
        with xr.open_dataset(filepath, group='product') as ds_prod:
            no2_raw = ds_prod['vertical_column_troposphere']
            q_flag = ds_prod['main_data_quality_flag']
            
            # Mask out bad pixels (keep only flag == 0 for highest quality)
            no2_clean = no2_raw.where(q_flag == 0)
            
            # Re-attach coordinates
            no2_clean = no2_clean.assign_coords({
                'latitude': lats,
                'longitude': lons,
                'time': times
            })

            # Perform spatial interpolation onto our 1x1 km master grid
            no2_regridded = no2_clean.interp(
                latitude=lat_master, 
                longitude=lon_master, 
                method='linear'
            )
            
            # Rename coordinates to match standard lat/lon
            no2_regridded = no2_regridded.rename({'latitude': 'lat', 'longitude': 'lon'})
            
            # Return dataset cleanly loaded into memory
            return no2_regridded.to_dataset(name='NO2_column').load()
            
    except Exception as e:
        print(f"  ⚠️ Skipping {filepath.name} due to read error: {e}")
        return None

# 2. Gather all files and group them by Year-Month
print("2. Scanning archive for NetCDF4 satellite files...")
all_files = sorted(list(tempo_dir.rglob('*.nc4')))
print(f"Found {len(all_files)} total files. Grouping by month...")

# Organize files into a dictionary: {'2023_08': [file1, file2...], ...}
files_by_month = {}
for fp in all_files:
    # TEMPO filenames look like: TEMPO_NO2_L3_V03_20230802T151249Z...
    # Extract the '202308' timestamp part
    parts = fp.name.split('_')
    for p in parts:
        if len(p) >= 6 and p[:4].isdigit() and p[0] == '2':
            ym_key = f"{p[:4]}_{p[4:6]}"
            files_by_month.setdefault(ym_key, []).append(fp)
            break

# 3. Process month-by-month
print(f"\n3. Starting batch processing across {len(files_by_month)} months...\n" + "="*50)

for ym_key in sorted(files_by_month.keys()):
    out_path = output_dir / f"tempo_no2_california_1x1km_{ym_key}.nc"
    
    if out_path.exists():
        print(f"⏭️ Skipping {ym_key}, processed file already exists.")
        continue
        
    month_files = files_by_month[ym_key]
    print(f"\n🗓️ Processing Month: {ym_key} ({len(month_files)} hourly scans)...")
    
    daily_datasets = []
    for i, fp in enumerate(month_files, 1):
        if i % 50 == 0 or i == len(month_files):
            print(f"   ...regridded {i}/{len(month_files)} files for {ym_key}")
        
        ds_hourly = load_and_regrid_single_file(fp)
        if ds_hourly is not None:
            daily_datasets.append(ds_hourly)
            
    if not daily_datasets:
        print(f"  ⚠️ No valid data extracted for {ym_key}. Skipping.")
        continue
        
    print(f"   Concatenating {len(daily_datasets)} hourly timestamps along time axis...")
    # Concatenate all hourly slices for this month into one unified time-series
    ds_month = xr.concat(daily_datasets, dim='time').sortby('time')
    
    # Add metadata
    ds_month.attrs['title'] = f"TEMPO V03 Regridded Tropospheric NO2 Column ({ym_key})"
    ds_month.attrs['resolution'] = "1 km (0.009 deg)"
    ds_month.attrs['units'] = "molecules/cm^2"
    
    print(f"   Writing monthly file to disk: {out_path.name}...")
    ds_month.to_netcdf(out_path, engine='netcdf4')
    
    # Explicitly clear RAM before starting the next month
    del daily_datasets, ds_month
    gc.collect()

print("\n🎉 All 5,760 TEMPO files successfully processed and aligned to the Master Grid!")