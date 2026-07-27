import xarray as xr
import rioxarray as rxr
from rasterio.merge import merge
import rasterio
import numpy as np
from pathlib import Path

# 1. Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
usgs_dir = BASE_DIR / 'data' / 'raw' / 'usgs'
globpop_dir = BASE_DIR / 'data' / 'raw' / 'globpop'
output_dir = BASE_DIR / 'data' / 'processed'
output_dir.mkdir(parents=True, exist_ok=True)

# 2. Define our standard 1 km x 1 km California Master Grid (EPSG:4326)
print("1. Initializing Master California Target Grid...")
lat_master = np.arange(32.5, 42.5, 0.009)
lon_master = np.arange(-124.5, -114.0, 0.009)

# Create an empty dummy xarray DataArray representing our exact target grid
target_grid = xr.DataArray(
    np.zeros((len(lat_master), len(lon_master))),
    coords={'y': lat_master, 'x': lon_master},
    dims=('y', 'x')
).rio.write_crs("EPSG:4326")

def process_raster_collection(folder_path, file_extension, var_name, output_filename, resampling_method="bilinear"):
    """
    Finds all raster tiles in a folder, mosaics them together, 
    reprojects to EPSG:4326, and resamples to the 1x1 km master grid.
    """
    print(f"\n--- Processing {var_name} from {folder_path.name} ---")
    files = sorted(list(folder_path.glob(f'*{file_extension}')))
    
    if not files:
        print(f"⚠️ No files found with extension {file_extension} in {folder_path}. Skipping.")
        return

    print(f"Found {len(files)} tiles. Mosaicing (stitching) together...")
    
    # Open all tiles using rasterio to perform a spatial merge
    src_files_to_mosaic = [rasterio.open(fp) for fp in files]
    mosaic_data, out_trans = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()
    
    # Close open file handles
    for src in src_files_to_mosaic:
        src.close()
        
    # Update metadata for the stitched raster
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic_data.shape[1],
        "width": mosaic_data.shape[2],
        "transform": out_trans
    })
    
    # Save temporary stitched GeoTIFF to memory/temp so rioxarray can ingest it cleanly
    temp_tif = output_dir / f"temp_stitched_{var_name}.tif"
    with rasterio.open(temp_tif, "w", **out_meta) as dest:
        dest.write(mosaic_data)
        
    print(f"Stitching complete! Resampling onto the 1 km Target Grid...")
    
    # Load the stitched mosaic into rioxarray
    rds = rxr.open_rasterio(temp_tif, masked=True).squeeze()
    
    # Reproject and match exact bounds and resolution of our master target grid
    resample_enum = rasterio.enums.Resampling.bilinear if resampling_method == "bilinear" else rasterio.enums.Resampling.sum
    regridded = rds.rio.reproject_match(target_grid, resampling=resample_enum)
    
    # Clean up dimensions and names for NetCDF consistency
    regridded.name = var_name
    regridded = regridded.rename({'y': 'lat', 'x': 'lon'})
    regridded.attrs['units'] = "meters" if var_name == "elevation" else "count/density"
    
    # Save final processed NetCDF
    out_path = output_dir / output_filename
    regridded.to_netcdf(out_path)
    
    # Remove temporary stitched GeoTIFF
    temp_tif.unlink(missing_ok=True)
    
    print(f"✅ Successfully saved {var_name} to {out_path}")

# 3. Execute Processing for USGS Elevation (using Bilinear interpolation)
process_raster_collection(usgs_dir, ".tif", "elevation", "usgs_elevation_1x1km.nc", resampling_method="bilinear")

# 4. Execute Processing for GlobPOP (using Bilinear for density or Sum for raw counts)
process_raster_collection(globpop_dir, ".tiff", "population_density", "globpop_1x1km.nc", resampling_method="bilinear")

print("\n🎉 All static raster layers stitched and aligned to the Master Grid!")