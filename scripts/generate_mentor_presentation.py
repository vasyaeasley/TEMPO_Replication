import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import xarray as xr

# 1. Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
raw_dir = BASE_DIR / 'data' / 'raw' / 'ecmwf'
regridded_dir = BASE_DIR / 'data' / 'processed' / 'era5_monthly_regridded'
tempo_dir = BASE_DIR / 'data' / 'processed' / 'tempo_monthly'
output_dir = BASE_DIR / 'data' / 'processed'

print('1. Locating cached August files to bypass the 300GB master lock...')
raw_files = sorted([
    fp
    for fp in raw_dir.glob('*.nc')
    if 'test' not in fp.name.lower() and not fp.name.startswith('.')
])
regridded_files = sorted(list(regridded_dir.glob('*.nc')))
tempo_files = sorted(list(tempo_dir.glob('*.nc')))

if not raw_files or not regridded_files or not tempo_files:
  raise FileNotFoundError(
      'Missing required monthly files in cache! Make sure earlier steps ran.'
  )

# Load August files specifically
ds_raw = xr.open_dataset(raw_files[0])
ds_regridded = xr.open_dataset(regridded_files[0])
ds_tempo = xr.open_dataset(tempo_files[0])

# --- Standardize ECMWF Naming Across Both Weather Files ---
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

# Align Raw Longitudes to Western Hemisphere
lon_coord = 'longitude' if 'longitude' in ds_raw.coords else 'lon'
lat_coord = 'latitude' if 'latitude' in ds_raw.coords else 'lat'
if ds_raw[lon_coord].max() > 180:
  ds_raw = ds_raw.assign_coords(
      {lon_coord: (((ds_raw[lon_coord] + 180) % 360) - 180)}
  )
ds_raw = ds_raw.sortby([lat_coord, lon_coord]).rename(
    {lat_coord: 'lat', lon_coord: 'lon'}
)
ds_raw_cali = ds_raw.sel(
    lat=slice(32.5, 42.5)
    if ds_raw['lat'][0] < ds_raw['lat'][-1]
    else slice(42.5, 32.5),
    lon=slice(-124.5, -114.0),
)
# ------------------------------------------------------------

# 2. Find High-Noon Overpass for Clear Satellite View
print('2. Finding optimal High-Noon afternoon overpass...')
valid_counts = ds_tempo['NO2_column'].isel(time=slice(0, 100)).count(dim=['lat', 'lon'])
best_idx = valid_counts.argmax().item()
sample_time = ds_tempo['time'].isel(time=best_idx).values
print(f'   -> Locked onto afternoon overpass: {str(sample_time)[:19]} UTC')

# Slice all three datasets at this exact timestamp
raw_slice = ds_raw_cali.sel(time=sample_time, method='nearest')
regrid_slice = ds_regridded.sel(time=sample_time, method='nearest')
tempo_slice = ds_tempo['NO2_column'].sel(time=sample_time, method='nearest')

raw_wind = np.sqrt(raw_slice['10m_u_wind'] ** 2 + raw_slice['10m_v_wind'] ** 2)
regrid_wind = np.sqrt(
    regrid_slice['10m_u_wind'] ** 2 + regrid_slice['10m_v_wind'] ** 2
)

lons = np.asarray(tempo_slice['lon'].values, dtype=float)
lats = np.asarray(tempo_slice['lat'].values, dtype=float)

# ==============================================================================
# FIGURE 1: WHY REGRIDDING MATTERS (Raw vs. Regridded)
# ==============================================================================
print('3. Generating Figure 1: Raw vs. Regridded Comparison...')
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
fig1.suptitle(
    f'Why Regridding is Essential: Raw (~25 km) vs. Digital Twin (~1 km)\nTimestamp: {str(sample_time)[:19]} UTC',
    fontsize=16,
    y=0.96,
)

im0 = axes1[0, 0].pcolormesh(
    raw_slice['lon'],
    raw_slice['lat'],
    raw_slice['2m_temperature'].values - 273.15,
    cmap='inferno',
    shading='nearest',
)
axes1[0, 0].set_title(
    'RAW ERA5 Temperature (~25 km / 0.25° Pixels)', fontsize=13, fontweight='bold'
)
plt.colorbar(im0, ax=axes1[0, 0], label='°C')

im1 = axes1[0, 1].pcolormesh(
    lons,
    lats,
    regrid_slice['2m_temperature'].values - 273.15,
    cmap='inferno',
    shading='auto',
)
axes1[0, 1].set_title(
    'REGRIDDED Digital Twin Temperature (~1 km Grid)',
    fontsize=13,
    fontweight='bold',
    color='darkblue',
)
plt.colorbar(im1, ax=axes1[0, 1], label='°C')

im2 = axes1[1, 0].pcolormesh(
    raw_slice['lon'],
    raw_slice['lat'],
    raw_wind.values,
    cmap='viridis',
    shading='nearest',
)
axes1[1, 0].set_title('RAW ERA5 Wind Speed (~25 km Pixels)', fontsize=13)
plt.colorbar(im2, ax=axes1[1, 0], label='m/s')

im3 = axes1[1, 1].pcolormesh(
    lons, lats, regrid_wind.values, cmap='viridis', shading='auto'
)
axes1[1, 1].set_title('REGRIDDED Digital Twin Wind Speed (~1 km Grid)', fontsize=13)
plt.colorbar(im3, ax=axes1[1, 1], label='m/s')

for ax in axes1.flat:
  ax.set_aspect('equal')
  ax.grid(True, linestyle='--', alpha=0.5, color='gray')
  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')
  ax.set_xlim([-124.5, -114.0])
  ax.set_ylim([32.5, 42.5])

plt.tight_layout()
out_fig1 = output_dir / 'mentor_fig1_why_regridding_matters.png'
plt.savefig(out_fig1, dpi=300, bbox_inches='tight')

# ==============================================================================
# FIGURE 2: THE DIGITAL TWIN PREVIEW (4-Quadrant Alignment)
# ==============================================================================
print('4. Generating Figure 2: Complete 4-Quadrant Digital Twin Alignment...')
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)
fig2.suptitle(
    f'Digital Twin Layer Alignment Preview (August Cache)\nTimestamp: {str(sample_time)[:19]} UTC (High Noon Overpass)',
    fontsize=16,
    y=0.96,
)

im_a = axes2[0, 0].pcolormesh(
    lons,
    lats,
    tempo_slice.values,
    cmap='Spectral_r',
    shading='auto',
    vmin=0,
    vmax=1e16,
)
axes2[0, 0].set_title(
    'TEMPO Tropospheric $\mathrm{NO}_2$ Column (~1 km)',
    fontsize=13,
    fontweight='bold',
    color='darkred',
)
plt.colorbar(im_a, ax=axes2[0, 0], label='$\mathrm{molecules/cm}^2$')

im_b = axes2[0, 1].pcolormesh(
    lons,
    lats,
    regrid_slice['2m_temperature'].values - 273.15,
    cmap='inferno',
    shading='auto',
)
axes2[0, 1].set_title(
    'ERA5 2m Temperature (~1 km Digital Twin Grid)', fontsize=13
)
plt.colorbar(im_b, ax=axes2[0, 1], label='°C')

im_c = axes2[1, 0].pcolormesh(
    lons, lats, regrid_wind.values, cmap='viridis', shading='auto'
)
axes2[1, 0].set_title('ERA5 10m Wind Speed (~1 km Digital Twin Grid)', fontsize=13)
plt.colorbar(im_c, ax=axes2[1, 0], label='m/s')

im_d = axes2[1, 1].pcolormesh(
    lons,
    lats,
    regrid_slice['surface_pressure'].values / 100,
    cmap='terrain',
    shading='auto',
)
axes2[1, 1].set_title(
    'ERA5 Surface Pressure (Topographic Proxy)', fontsize=13
)
plt.colorbar(im_d, ax=axes2[1, 1], label='hPa')

for ax in axes2.flat:
  ax.set_aspect('equal')
  ax.grid(True, linestyle='--', alpha=0.4, color='gray')
  ax.set_xlabel('Longitude')
  ax.set_ylabel('Latitude')
  ax.set_xlim([-124.5, -114.0])
  ax.set_ylim([32.5, 42.5])

plt.tight_layout()
out_fig2 = output_dir / 'mentor_fig2_digital_twin_preview.png'
plt.savefig(out_fig2, dpi=300, bbox_inches='tight')

print('\n🎯 BOTH MENTOR PRESENTATION MAPS GENERATED SUCCESSFULLY!')
print(f'   1. {out_fig1}')
print(f'   2. {out_fig2}')