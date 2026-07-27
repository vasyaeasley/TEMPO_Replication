from pathlib import Path
import netCDF4 as nc

# 1. Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
processed_dir = BASE_DIR / 'data' / 'processed'

master_file = processed_dir / 'era5_california_1x1km_master.nc'
pop_file = processed_dir / 'globpop_1x1km.nc'
elev_file = processed_dir / 'usgs_elevation_1x1km.nc'

print('🔍 STATIC GRID ALIGNMENT SANITY CHECK 🔍')
print('=' * 55)

# 2. Get Master Grid Dimensions as our Baseline
with nc.Dataset(master_file, 'r') as m_ds:
  lat_key = 'lat' if 'lat' in m_ds.variables else 'latitude'
  lon_key = 'lon' if 'lon' in m_ds.variables else 'longitude'
  target_lat = m_ds.variables[lat_key].shape[0]
  target_lon = m_ds.variables[lon_key].shape[0]

print(f'🎯 Master Weather Grid Target: [{target_lat} rows x {target_lon} cols]')
print('=' * 55)

# 3. Inspect Static Files
for name, fp in [
    ('Population Density', pop_file),
    ('USGS Topography Elevation', elev_file),
]:
  print(f'\n📂 Checking {name}...')
  print(f'   File: {fp.name}')

  if not fp.exists():
    print('   ❌ ERROR: File not found!')
    continue

  with nc.Dataset(fp, 'r') as ds:
    # Find spatial dimensions
    dims = {k: len(v) for k, v in ds.dimensions.items()}
    vars_list = list(ds.variables.keys())

    # Get exact row/col counts
    row_count = next((v for k, v in dims.items() if 'lat' in k.lower()), 'N/A')
    col_count = next((v for k, v in dims.items() if 'lon' in k.lower()), 'N/A')

    print(f'   Dimensions Found: [{row_count} rows x {col_count} cols]')
    print(f'   Variables:        {vars_list}')

    # Verify alignment
    if row_count == target_lat and col_count == target_lon:
      print('   ✅ PERFECT MATCH! Ready for multi-modal GPU fusion!')
    else:
      print(
          '   ⚠️ MISMATCH DETECTED: Needs a quick 1-line xarray regridding before training.'
      )

print('\n🎯 Inspection Complete!')