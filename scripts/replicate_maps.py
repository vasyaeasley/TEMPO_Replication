import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Safe import for GeoPandas (standard for spatial Python)
try:
  import geopandas as gpd

  HAS_GPD = True
except ImportError:
  HAS_GPD = False
  print(
      '⚠️ Warning: GeoPandas not installed. Road network overlay will be skipped.'
  )

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION (Fixed for data/raw/ directory structure!)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / 'data' / 'raw' / 'epa_from_internet_daily'
OUTPUT_DIR = BASE_DIR / 'models'
OUTPUT_DIR.mkdir(exist_ok=True)

# Bounding Box for Los Angeles Basin (from Slide #2)
LAT_MIN, LAT_MAX = 33.6, 34.4
LON_MIN, LON_MAX = -118.7, -117.2

print('🌍 STARTING LA BASIN EPA MONITOR MAP REPLICATION (WITH HIGHWAYS) 🌍')
print('=' * 70)

# ==============================================================================
# 2. LOAD & FILTER 2024 NO2 MONITOR DATA
# ==============================================================================
no2_file = RAW_DIR / 'daily_42602_2024.csv'
if not no2_file.exists():
  # Fallback check in case script is run from a different root
  no2_file = BASE_DIR / 'raw' / 'epa_from_internet_daily' / 'daily_42602_2024.csv'
  if not no2_file.exists():
    raise FileNotFoundError(
        f'Missing NO2 data file at: {RAW_DIR / "daily_42602_2024.csv"}'
    )

print(f'⏳ Loading 2024 NO2 Daily Summary Data...')
df_no2 = pd.read_csv(no2_file, low_memory=False)

# Filter strictly to the Los Angeles Basin coordinate window
mask_bbox = (
    (df_no2['Latitude'] >= LAT_MIN)
    & (df_no2['Latitude'] <= LAT_MAX)
    & (df_no2['Longitude'] >= LON_MIN)
    & (df_no2['Longitude'] <= LON_MAX)
)
df_la = df_no2[mask_bbox].copy()

# Create a unique Site ID (State-County-Site)
df_la['Site_ID'] = (
    df_la['State Code'].astype(str).str.zfill(2)
    + '-'
    + df_la['County Code'].astype(str).str.zfill(3)
    + '-'
    + df_la['Site Num'].astype(str).str.zfill(4)
)

# Aggregate by site to get coordinates, name, and 2024 Maximum Daily Mean
sites_df = (
    df_la.groupby('Site_ID')
    .agg({
        'Latitude': 'first',
        'Longitude': 'first',
        'Local Site Name': 'first',
        'Arithmetic Mean': 'max',
    })
    .reset_index()
)

sites_df['Local Site Name'] = sites_df['Local Site Name'].fillna(
    sites_df['Site_ID']
)

# Convert ppm to ppb if values are stored in ppm (< 5.0)
if sites_df['Arithmetic Mean'].max() < 5.0:
  sites_df['Max_NO2_ppb'] = sites_df['Arithmetic Mean'] * 1000.0
else:
  sites_df['Max_NO2_ppb'] = sites_df['Arithmetic Mean']

print(f'✅ Found {len(sites_df)} valid NO2 monitoring sites in the LA Basin!')

# ==============================================================================
# 3. IDENTIFY COMPLETE METEOROLOGY SITES
# ==============================================================================
met_files = [
    RAW_DIR / 'daily_TEMP_2024.csv',
    RAW_DIR / 'daily_WIND_2024.csv',
    RAW_DIR / 'daily_PRESS_2024.csv',
    RAW_DIR / 'daily_RH_DP_2024.csv',
]

print('⏳ Checking Meteorological Coverage across files...')
met_site_ids = set(sites_df['Site_ID'])

for m_file in met_files:
  if not m_file.exists():
    m_file = (
        BASE_DIR / 'raw' / 'epa_from_internet_daily' / m_file.name
    )  # fallback
  if m_file.exists():
    df_m = pd.read_csv(m_file, usecols=['State Code', 'County Code', 'Site Num'])
    m_ids = set(
        df_m['State Code'].astype(str).str.zfill(2)
        + '-'
        + df_m['County Code'].astype(str).str.zfill(3)
        + '-'
        + df_m['Site Num'].astype(str).str.zfill(4)
    )
    met_site_ids = met_site_ids.intersection(m_ids)
  else:
    print(f'   ⚠️ Warning: Could not find {m_file.name}, skipping check.')

sites_df['Has_Met'] = sites_df['Site_ID'].isin(met_site_ids)
n_met = sites_df['Has_Met'].sum()
n_no2_only = len(sites_df) - n_met
print(
    f'   -> Complete Meteorology Sites: {n_met} | NO2-Only Sites: {n_no2_only}'
)

# ==============================================================================
# 4. LOAD & CROP HIGHWAY / ROAD NETWORK (From your raw/ folder!)
# ==============================================================================
gdf_la_roads = None
if HAS_GPD:
  # Check paths based on your file tree screenshot
  possible_road_paths = [
      BASE_DIR
      / 'data'
      / 'raw'
      / 'highway_map_california'
      / 'National_Highway_System.geojson',
      BASE_DIR
      / 'raw'
      / 'highway_map_california'
      / 'National_Highway_System.geojson',
      BASE_DIR
      / 'data'
      / 'raw'
      / 'grip'
      / 'National_Highway_System.geojson',
      BASE_DIR
      / 'data'
      / 'raw'
      / 'traffic'
      / 'Traffic_Volumes_AADT.geojson',  # fallback to traffic corridor lines if needed
  ]

  road_path = next((p for p in possible_road_paths if p.exists()), None)

  if road_path:
    print(f'⏳ Loading Highway Network from: {road_path.name}...')
    try:
      gdf_roads = gpd.read_file(road_path)
      # Ensure WGS84 coordinate system (EPSG:4326)
      if gdf_roads.crs is not None and gdf_roads.crs.to_string() != 'EPSG:4326':
        gdf_roads = gdf_roads.to_crs('EPSG:4326')

      # Fast spatial crop using bounding box slice (.cx)
      gdf_la_roads = gdf_roads.cx[LON_MIN:LON_MAX, LAT_MIN:LAT_MAX].copy()
      print(
          f'✅ Extracted {len(gdf_la_roads)} highway/road segments inside the LA Basin!'
      )
    except Exception as e:
      print(f'   ⚠️ Could not process road network file: {e}')
  else:
    print(
        '   ⚠️ No highway GeoJSON found in data/raw/highway_map_california/. Proceeding without roads.'
    )
print('=' * 70)


# Helper function to plot roads cleanly on any axis
def plot_highways(ax, transform_kw):
  if gdf_la_roads is not None and not gdf_la_roads.empty:
    gdf_la_roads.plot(
        ax=ax,
        color='#555555',  # Charcoal Grey
        linewidth=0.8,
        alpha=0.65,
        zorder=2,  # Layered above land (0/1), below scatter dots (3)
        label='Major Highways / Roads',
        **transform_kw,
    )


# ==============================================================================
# 5. PLOT MAP 1: METEOROLOGY STATUS (REPLICATING SLIDE #3)
# ==============================================================================
print('🎨 Generating Map 1: Monitoring Sites, Meteorology & Highways...')
fig1, ax1 = plt.subplots(figsize=(16, 9.5))

try:
  import cartopy.crs as ccrs
  import cartopy.feature as cfeature

  plt.close(fig1)
  fig1 = plt.figure(figsize=(16, 9.5))
  ax1 = fig1.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
  ax1.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
  ax1.add_feature(cfeature.OCEAN, facecolor='#d5e8f7', zorder=0)
  ax1.add_feature(cfeature.LAND, facecolor='#f4f3ed', zorder=0)
  ax1.add_feature(cfeature.COASTLINE, linewidth=1.2, zorder=1)
  ax1.add_feature(cfeature.BORDERS, linestyle=':', zorder=1)
  transform_arg = {'transform': ccrs.PlateCarree()}
except ImportError:
  ax1.set_xlim(LON_MIN, LON_MAX)
  ax1.set_ylim(LAT_MIN, LAT_MAX)
  ax1.set_facecolor('#f4f3ed')
  transform_arg = {}

ax1.grid(True, linestyle='--', alpha=0.5, color='gray', zorder=1)

# Overlay Highways & Roads (zorder=2)
plot_highways(ax1, transform_arg)

# Scatter plot for NO2 Only vs Complete Met (zorder=3)
met_mask = sites_df['Has_Met']
ax1.scatter(
    sites_df.loc[~met_mask, 'Longitude'],
    sites_df.loc[~met_mask, 'Latitude'],
    color='#2ecc71',
    s=90,
    edgecolor='black',
    zorder=3,
    label=f'NO₂ only (n={n_no2_only})',
    **transform_arg,
)
ax1.scatter(
    sites_df.loc[met_mask, 'Longitude'],
    sites_df.loc[met_mask, 'Latitude'],
    color='#1f77b4',
    s=90,
    edgecolor='black',
    zorder=3,
    label=f'Complete meteorology (n={n_met})',
    **transform_arg,
)

# Annotate Site Names (zorder=4)
for _, row in sites_df.iterrows():
  color = '#1f77b4' if row['Has_Met'] else '#2ecc71'
  ax1.annotate(
      row['Local Site Name'],
      xy=(row['Longitude'], row['Latitude']),
      xytext=(0, 10),
      textcoords='offset points',
      ha='center',
      fontsize=8,
      fontweight='bold',
      color=color,
      bbox=dict(
          boxstyle='round,pad=0.3',
          facecolor='white',
          edgecolor=color,
          alpha=0.9,
      ),
      zorder=4,
      **transform_arg,
  )

ax1.set_title(
    'EPA NO₂ Monitoring Sites & Highway Network - Los Angeles Basin 2024',
    fontsize=14,
    fontweight='bold',
    pad=15,
)
ax1.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='gray')

summary_text = (
    f'Complete Meteorology Sites: {n_met}\n'
    f'NO₂-Only Sites: {n_no2_only}\n'
    f'Total Sites: {len(sites_df)}\n\n'
    'Parameters:\n • NO₂\n • Temperature\n • Pressure\n • Humidity/Dew Point\n • Wind'
)
ax1.text(
    0.02,
    0.03,
    summary_text,
    transform=ax1.transAxes,
    fontsize=9,
    verticalalignment='bottom',
    bbox=dict(
        boxstyle='square,pad=0.6', facecolor='white', edgecolor='gray', alpha=0.9
    ),
    zorder=5,
)

out_file1 = OUTPUT_DIR / 'la_basin_epa_sites_meteorology_with_roads.png'
plt.savefig(out_file1, dpi=300, bbox_inches='tight')
print(f'   -> Saved Map 1 to: {out_file1}')

# ==============================================================================
# 6. PLOT MAP 2: MAXIMUM DAILY MEAN VALUES (REPLICATING SLIDE #4)
# ==============================================================================
print('🎨 Generating Map 2: Maximum Daily Mean Values & Highway Network...')
fig2, ax2 = plt.subplots(figsize=(16, 9.5))

try:
  import cartopy.crs as ccrs
  import cartopy.feature as cfeature

  plt.close(fig2)
  fig2 = plt.figure(figsize=(16, 9.5))
  ax2 = fig2.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
  ax2.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())
  ax2.add_feature(cfeature.OCEAN, facecolor='#d5e8f7', zorder=0)
  ax2.add_feature(cfeature.LAND, facecolor='#f4f3ed', zorder=0)
  ax2.add_feature(cfeature.COASTLINE, linewidth=1.2, zorder=1)
  ax2.add_feature(cfeature.BORDERS, linestyle=':', zorder=1)
except ImportError:
  ax2.set_xlim(LON_MIN, LON_MAX)
  ax2.set_ylim(LAT_MIN, LAT_MAX)
  ax2.set_facecolor('#f4f3ed')

ax2.grid(True, linestyle='--', alpha=0.5, color='gray', zorder=1)

# Overlay Highways & Roads (zorder=2)
plot_highways(ax2, transform_arg)

# Scatter plot colored by Maximum Daily Mean NO2 (zorder=3)
sc = ax2.scatter(
    sites_df['Longitude'],
    sites_df['Latitude'],
    c=sites_df['Max_NO2_ppb'],
    cmap='RdYlBu_r',  # Blue=Low, Red=High
    s=100,
    edgecolor='black',
    vmin=20,
    vmax=55,
    zorder=3,
    **transform_arg,
)

cbar = fig2.colorbar(
    sc, ax=ax2, orientation='horizontal', pad=0.08, shrink=0.6, aspect=35
)
cbar.set_label('Maximum NO₂ Daily Mean (ppb)', fontsize=11, fontweight='bold')

# Annotate Site Names with their Max Value (zorder=4)
for _, row in sites_df.iterrows():
  label_text = f"{row['Local Site Name']}\n{row['Max_NO2_ppb']:.1f} ppb"
  ax2.annotate(
      label_text,
      xy=(row['Longitude'], row['Latitude']),
      xytext=(0, 10),
      textcoords='offset points',
      ha='center',
      fontsize=7.5,
      fontweight='bold',
      color='black',
      bbox=dict(
          boxstyle='round,pad=0.3',
          facecolor='white',
          edgecolor='gray',
          alpha=0.85,
      ),
      zorder=4,
      **transform_arg,
  )

ax2.set_title(
    'EPA NO₂ Monitoring Sites vs. Major Highway Corridors\nMaximum Daily Mean'
    ' Values 2024 - Los Angeles Basin',
    fontsize=14,
    fontweight='bold',
    pad=15,
)
if gdf_la_roads is not None and not gdf_la_roads.empty:
  ax2.legend(
      loc='upper left', frameon=True, facecolor='white', edgecolor='gray'
  )

max_site = sites_df.loc[sites_df['Max_NO2_ppb'].idxmax()]
min_val = sites_df['Max_NO2_ppb'].min()
mean_val = sites_df['Max_NO2_ppb'].mean()

stats_text = (
    f'Total Sites: {len(sites_df)}\n\n'
    f'Maximum NO₂ Values:\n'
    f'Highest: {max_site["Max_NO2_ppb"]:.2f} ppb\n'
    f'Lowest: {min_val:.2f} ppb\n'
    f'Mean: {mean_val:.2f} ppb\n\n'
    f'Site with Highest Max:\n'
    f'{max_site["Local Site Name"]} ({max_site["Max_NO2_ppb"]:.2f} ppb)'
)
ax2.text(
    0.02,
    0.03,
    stats_text,
    transform=ax2.transAxes,
    fontsize=8.5,
    verticalalignment='bottom',
    bbox=dict(
        boxstyle='square,pad=0.6', facecolor='white', edgecolor='gray', alpha=0.9
    ),
    zorder=5,
)

source_text = (
    'Data Sources:\n • EPA Air Quality System (2024)\n • National Highway System'
    ' (Caltrans)'
)
ax2.text(
    0.98,
    0.97,
    source_text,
    transform=ax2.transAxes,
    fontsize=8.5,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(
        boxstyle='square,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9
    ),
    zorder=5,
)

out_file2 = OUTPUT_DIR / 'la_basin_epa_sites_max_no2_with_roads.png'
plt.savefig(out_file2, dpi=300, bbox_inches='tight')
print(f'   -> Saved Map 2 to: {out_file2}')

print('\n🎯 REPLICATION WITH HIGHWAY OVERLAY SUCCESSFULLY FINISHED!')