import gc
from pathlib import Path
import numpy as np
import xarray as xr

# 1. Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
era5_dir = BASE_DIR / 'data' / 'raw' / 'ecmwf'
output_dir = BASE_DIR / 'data' / 'processed'

# Create a temporary folder for monthly interpolated files (just like TEMPO!)
monthly_out_dir = output_dir / 'era5_monthly_regridded'
monthly_out_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / 'era5_california_1x1km_master.nc'

print('1. Defining the 1x1 km Master Grid over California...')
lat_master = np.arange(32.5, 42.5, 0.009)
lon_master = np.arange(-124.5, -114.0, 0.009)

# Strictly filter out leftover test files and hidden OS files
all_nc_files = sorted(list(era5_dir.glob('*.nc')))
era5_files = [
    fp
    for fp in all_nc_files
    if 'test' not in fp.name.lower() and not fp.name.startswith('.')
]

if not era5_files:
  raise FileNotFoundError('No valid monthly ERA5 NetCDF files found!')

print(
    f'2. Processing {len(era5_files)} clean files Month-by-Month (Divide &'
    ' Conquer)...'
)
print('   This prevents RAM spikes and bypasses Dask memory limits entirely!\n')

for i, fp in enumerate(era5_files, 1):
  # Define output filename for this specific month
  month_out_path = monthly_out_dir / f'regridded_{fp.name}'

  if month_out_path.exists():
    print(
        f'   [{i}/{len(era5_files)}] ⏭️ Skipping {fp.name}, already regridded!'
    )
    continue

  print(f'   [{i}/{len(era5_files)}] Interpolating {fp.name} in RAM...')

  # Load single month directly into memory
  with xr.open_dataset(fp, engine='netcdf4') as ds_month:
    # Drop nuisance string metadata that causes object dtype errors
    ds_month = ds_month.drop_vars(['expver', 'number'], errors='ignore')

    lat_coord = 'latitude' if 'latitude' in ds_month.coords else 'lat'
    lon_coord = 'longitude' if 'longitude' in ds_month.coords else 'lon'

    # Convert longitudes from [0, 360] to [-180, 180]
    if ds_month[lon_coord].max() > 180:
      ds_month = ds_month.assign_coords(
          {lon_coord: (((ds_month[lon_coord] + 180) % 360) - 180)}
      )

    # Force strict ascending order to prevent NaN blanks
    ds_month = ds_month.sortby([lat_coord, lon_coord])

    # Execute fast linear spatial interpolation on just this single month
    ds_interp = ds_month.interp(
        {lat_coord: lat_master, lon_coord: lon_master},
        method='linear',
        kwargs={'fill_value': 'extrapolate'},
    ).compute()

    ds_interp = ds_interp.rename({lat_coord: 'lat', lon_coord: 'lon'})

    # Write this month directly to disk to lock in progress
    ds_interp.to_netcdf(month_out_path, engine='netcdf4')

  # Explicitly clear RAM before loading the next month!
  del ds_interp
  gc.collect()

print('\n3. All months pre-processed! Concatenating into Master Dataset...')
# Now that the heavy interpolation math is done, combining files takes seconds!
regridded_files = sorted(list(monthly_out_dir.glob('*.nc')))
ds_master = xr.open_mfdataset(
    regridded_files, combine='by_coords', parallel=False
)

ds_master.attrs['title'] = (
    'Regridded 1x1 km ERA5 Meteorology for TEMPO Digital Twin'
)
ds_master.attrs['resolution'] = '1 km (0.009 deg)'

print('4. Writing unified master dataset to disk...')
ds_master.to_netcdf(output_path, engine='netcdf4')

print(f'\n✅ Success! Master weather dataset saved to:\n{output_path}')