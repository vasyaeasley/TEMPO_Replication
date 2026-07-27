from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

# ==============================================================================
# 1. SETUP PATHS & LA BASIN BOUNDING BOX
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
processed_dir = BASE_DIR / "data" / "processed"
tempo_dir = processed_dir / "tempo_monthly"

# LA Basin Spatial Bounding Box
# Widened Bounding Box to zoom out and capture the full mountain boundaries
LAT_MIN, LAT_MAX = 32.4, 35.5    # Dropped South to San Diego border, raised North past mountains
LON_MIN, LON_MAX = -120.00, -114.00 # Expanded West out to sea, East all the way past Riverside

tempo_files = sorted(list(tempo_dir.glob("*.nc")))
if not tempo_files:
    raise FileNotFoundError("❌ No TEMPO files found in tempo_monthly/")
target_file = tempo_files[0]

print("=" * 65)
print("🗺️ GENERATING CARTOPY GEOGRAPHIC MAP FOR LA BASIN 🗺️")
print("=" * 65)

# ==============================================================================
# 2. EXTRACT SPATIAL MATRICES & LAT/LON COORDINATES
# ==============================================================================
with nc.Dataset(target_file, "r") as ds:
    no2_raw = np.ma.filled(ds.variables["NO2_column"][:].astype("float64"), np.nan)
    lats = ds.variables["lat"][:] if "lat" in ds.variables else ds.variables["latitude"][:]
    lons = ds.variables["lon"][:] if "lon" in ds.variables else ds.variables["longitude"][:]

if lats.ndim == 1:
    lon_grid, lat_grid = np.meshgrid(lons, lats)
else:
    lat_grid, lon_grid = lats, lons

# Select a clean afternoon time slice index
time_idx = min(12, no2_raw.shape[0] - 1)
no2_slice = no2_raw[time_idx, :, :] / 1e15

# Clean out missing values/cloud flags
no2_slice = np.where(no2_slice < 0, np.nan, no2_slice)

# ==============================================================================
# 3. PLOT WITH CARTOPY GEOGRAPHIC LAYERS
# ==============================================================================
plt.figure(figsize=(10, 8))

# Define map projection (PlateCarrée standard lat/lon projection)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=ccrs.PlateCarree())

# Plot the satellite data using the raw coordinate meshgrids
mesh = ax.pcolormesh(
    lon_grid,
    lat_grid,
    no2_slice,
    cmap="jet",
    transform=ccrs.PlateCarree(),
    vmin=np.nanpercentile(no2_slice, 5),
    vmax=np.nanpercentile(no2_slice, 95),
    alpha=0.85
)

# --- ADD GEOGRAPHIC CARTO FEATURES ---
# 1. Add high-resolution coastlines and ocean mask
ax.add_feature(cfeature.COASTLINE.with_scale("10m"), edgecolor="black", linewidth=1.5)
ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="none", edgecolor="none")

# 2. Add US County Borders (Crucial for identifying LA vs Orange County vs Inland Empire)
counties = cfeature.NaturalEarthFeature(
    category="cultural",
    name="admin_2_counties",
    scale="10m",
    facecolor="none",
    edgecolor="gray",
    linewidth=0.8,
    linestyle="--"
)
ax.add_feature(counties)

# 3. Add Gridlines with Latitude/Longitude labels
gl = ax.gridlines(draw_labels=True, linestyle=":", color="black", alpha=0.6)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": 10, "weight": "bold"}
gl.ylabel_style = {"size": 10, "weight": "bold"}

# Add Colorbar and Title
plt.colorbar(mesh, label="TEMPO NO₂ Column Density (10¹⁵ molec/cm²)", orientation="vertical", shrink=0.7)
plt.title("NASA TEMPO Satellite NO₂ Column\nGreater Los Angeles & Anaheim Air Basin Corridor", fontsize=13, weight="bold", pad=15)

# Save high-resolution figure
output_path = processed_dir / "la_basin_cartopy_map.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"✅ Mapping complete! High-resolution map saved to:\n   {output_path}")
print("=" * 65)