from pathlib import Path
import netCDF4 as nc

BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Point to the AirNow directory (following the symlink!)
airnow_dir = BASE_DIR / 'data' / 'raw' / 'epa' / 'AirNow'
if not airnow_dir.exists():
  # Fallback to a secondary local layout some setups use.
  airnow_dir = BASE_DIR / 'data' / 'raw' / 'AirNow'

print('🔍 SEARCHING FOR AIRNOW NETCDF FILES...')
nc_files = sorted(list(airnow_dir.glob('*.nc')))

if not nc_files:
  print('❌ No .nc files found in:', airnow_dir)
else:
  sample_file = nc_files[0]
  print(f'✅ Found {len(nc_files):,} AirNow NetCDF files!')
  print(f'📄 Inspecting Sample File: {sample_file.name}')
  print('=' * 60)

  with nc.Dataset(sample_file, 'r') as ds:
    print('📐 Dimensions:')
    for k, v in ds.dimensions.items():
      print(f'   - {k}: {len(v)}')

    print('\n📊 Variables:')
    for k, v in ds.variables.items():
      print(f'   - [{k}]: shape {v.shape} | dtype {v.dtype}')
      if hasattr(v, 'units'):
        print(f'       Units: {v.units}')
      if hasattr(v, 'long_name'):
        print(f'       Description: {v.long_name}')

print('=' * 60)
print('🎯 Ready to build the 14-Month Point-Coordinate Extraction Loop!')