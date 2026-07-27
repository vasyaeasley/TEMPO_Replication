import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import xarray as xr

# 1. Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
raw_dir = BASE_DIR / 'data' / 'raw' / 'ecmwf'
regridded_dir = BASE_DIR / 'data' / 'processed' / 'era5_monthly_regridded'
output_dir = BASE_DIR / 'data' / 'processed'

# 2. Find matching raw and regridded monthly files
raw_files = sorted([
    fp
    for fp in raw_dir.glob('*.nc')
    if 'test' not in fp.name.lower() and not fp.name.startswith('.')
])
regridded_files = sorted(list(regridded_dir.glob('*.nc')))

if not raw_files or not regridded_files:
  raise FileNotFoundError(
      'Could not find raw or regridded monthly files for comparison!'
  )

print(f'1. Loading Raw File: {raw_files[0].name}...')
ds_raw = xr.open_dataset(raw_files[0])

print(f'2. Loading Regridded File: {regridded_files[0].name}...')
ds_regridded = xr.open_dataset(regridded_files[0])

# --- THE FIX: Standardize BOTH datasets using a universal loop! ---
ecmwf_short_to_long = {
    'u10': '10m_u_wind',
    'v10': '10m_v_wind',
    't2m': '2m_temperature',
    'sp': 'surface_pressure',
    'blh': 'boundary_layer_height',
}

for name, ds in [('Raw', ds_raw), ('Regridded', ds_regridded)]:
  if 'valid_time' in ds.coords or 'valid_time' in ds.dims:
    ds = ds.rename({'valid_time': 'time'})
  rename_dict = {k: v for k, v in ecmwf_short_to_long.items() if k in ds.data_vars}
  if rename_dict:
    ds = ds.rename(rename_dict)

  if name == 'Raw':
    ds_raw = ds
  else:
    ds_regridded = ds
# ------------------------------------------------------------------

# --- Align Raw Longitudes to California ---
lon_coord = 'longitude' if 'longitude' in ds_raw.coords else 'lon'
lat_coord = 'latitude' if 'latitude' in ds_raw.coords else 'lat'

if ds_raw[lon_coord].max() > 180:
  ds_raw = ds_raw.assign_coords(
      {lon_coord: (((ds_raw[lon_coord] + 180) % 360) - 180)}
  )
ds_raw = ds_raw.sortby([lat_coord, lon_coord])
ds_raw = ds_raw.rename({lat_coord: 'lat', lon_coord: 'lon'})

# Slice raw data to California bounding box
ds_raw_cali = ds_raw.sel(
    lat=slice(32.5, 42.5)
    if ds_raw['lat'][0] < ds_raw['lat'][-1]
    else slice(42.5, 32.5),
    lon=slice(-124.5, -114.0),
)
# ------------------------------------------

# 3. Select a matching daytime timestamp halfway through the month
valid_times = ds_regridded['time'].dropna(dim='time', how='all').values
sample_time = valid_times[len(valid_times) // 2]

print(f'3. Slicing both datasets for timestamp: {str(sample_time)[:19]} UTC...')
raw_slice = ds_raw_cali.sel(time=sample_time, method='nearest')
regrid_slice = ds_regridded.sel(time=sample_time, method='nearest')

# Calculate wind speeds
raw_wind = np.sqrt(raw_slice['10m_u_wind'] ** 2 + raw_slice['10m_v_wind'] ** 2)
regrid_wind = np.sqrt(
    regrid_slice['10m_u_wind'] ** 2 + regrid_slice['10m_v_wind'] ** 2
)

# 4. Generate the Comparison Figure
print('4. Drawing side-by-side comparison maps...')
fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
fig.suptitle(
    f'Why Regridding is Essential: Raw (~25 km) vs. Digital Twin (~1 km)\nTimestamp: {str(sample_time)[:19]} UTC',
    fontsize=16,
    y=0.96,
)

# Plot A: Raw Temperature (Changed to shading='nearest' to fix TypeError!)
im0 = axes[0, 0].pcolormesh(
    raw_slice['lon'],
    raw_slice['lat'],
    raw_slice['2m_temperature'].values - 273.15,
    cmap='inferno',
    shading='nearest',
)
axes[0, 0].set_title(
    'RAW ERA5 Temperature (~25 km / 0.25° Pixels)', fontsize=13, fontweight='bold'
)
plt.colorbar(im0, ax=axes[0, 0], label='°C')

# Plot B: Regridded Temperature (Smooth 1 km)
im1 = axes[0, 1].pcolormesh(
    regrid_slice['lon'],
    regrid_slice['lat'],
    regrid_slice['2m_temperature'].values - 273.15,
    cmap='inferno',
    shading='auto',
)
axes[0, 1].set_title(
    'REGRIDDED Digital Twin Temperature (~1 km Grid)',
    fontsize=13,
    fontweight='bold',
    color='darkblue',
)
plt.colorbar(im1, ax=axes[0, 1], label='°C')

# Plot C: Raw Wind Speed (Changed to shading='nearest'!)
im2 = axes[1, 0].pcolormesh(
    raw_slice['lon'],
    raw_slice['lat'],
    raw_wind.values,
    cmap='viridis',
    shading='nearest',
)
axes[1, 0].set_title('RAW ERA5 Wind Speed (~25 km Pixels)', fontsize=13)
plt.colorbar(im2, ax=axes[1, 0], label='m/s')

# Plot D: Regridded Wind Speed (Smooth 1 km)
im3 = axes[1, 1].pcolormesh(
    regrid_slice['lon'],
    regrid_slice['lat'],
    regrid_wind.values,
    cmap='viridis',
    shading='auto',
)
axes[1, 1].set_title('REGRIDDED Digital Twin Wind Speed (~1 km Grid)', fontsize=13)
plt.colorbar(im3, ax=axes[1, 1], label='m/s')

# Format axes and gridlines
for ax in axes.flat:
  ax.set_aspect('equal')
  ax.grid(True, linestyle='--', alpha=0.5, color='gray')
  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')
  ax.set_xlim([-124.5, -114.0])
  ax.set_ylim([32.5, 42.5])

plt.tight_layout()

# 5. Save Output
out_image = output_dir / 'raw_vs_regridded_comparison.png'
plt.savefig(out_image, dpi=300, bbox_inches='tight')
print(f'\n✅ Comparison map saved successfully to:\n{out_image}')