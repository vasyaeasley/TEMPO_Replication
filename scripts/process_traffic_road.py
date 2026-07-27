"""Build 1x1km traffic-density and road-density rasters aligned to the master grid.

Two new static covariates (matching the IEEE paper's feature set):
  * traffic_aadt  : Caltrans Traffic Volumes (AADT) point layer, rasterized to the
                    1km grid (log1p of max AADT per cell) and Gaussian-smoothed so
                    each cell reflects nearby highway traffic intensity.
  * road_density  : GRIP4 global total road density (m/km^2) regridded to 1km.

Outputs (aligned to the existing globpop_1x1km.nc grid):
  data/processed/traffic_aadt_1x1km.nc
  data/processed/grip_road_density_1x1km.nc
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path

import netCDF4 as nc
import numpy as np
from scipy.ndimage import gaussian_filter

BASE_DIR = Path(__file__).resolve().parent.parent
processed = BASE_DIR / "data" / "processed"
grip_asc = BASE_DIR / "data" / "raw" / "grip" / "grip4_total_dens_m_km2.asc"

AADT_URL = (
    "https://caltrans-gis.dot.ca.gov/arcgis/rest/services/"
    "CHhighway/Traffic_AADT/FeatureServer/0/query"
)

# ---------------------------------------------------------------------------
# Target 1km grid (reuse the exact coords of an existing processed raster).
# ---------------------------------------------------------------------------
with nc.Dataset(processed / "globpop_1x1km.nc") as ds:
    glat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
    glon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
nlat, nlon = len(glat), len(glon)
lat_res = float(glat[1] - glat[0])
lon_res = float(glon[1] - glon[0])
print(f"Target grid: {nlat} x {nlon}  (lat_res={lat_res:.4f}, lon_res={lon_res:.4f})")


def _num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# 1. TRAFFIC — download all AADT points (paginated GeoJSON) and rasterize.
# ---------------------------------------------------------------------------
def fetch_aadt():
    points = []
    offset, page = 0, 2000
    while True:
        params = urllib.parse.urlencode({
            "where": "1=1",
            "outFields": "BACK_AADT,AHEAD_AADT",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": page,
        })
        with urllib.request.urlopen(f"{AADT_URL}?{params}", timeout=90) as resp:
            gj = json.loads(resp.read().decode("utf-8"))
        feats = gj.get("features", [])
        if not feats:
            break
        for ft in feats:
            geom = ft.get("geometry") or {}
            coords = geom.get("coordinates")
            if not coords:
                continue
            props = ft.get("properties", {})
            aadt = np.nanmax([_num(props.get("BACK_AADT")), _num(props.get("AHEAD_AADT"))])
            if np.isfinite(aadt) and aadt > 0:
                points.append((float(coords[0]), float(coords[1]), aadt))
        print(f"   fetched {offset + len(feats)} features...", flush=True)
        offset += len(feats)
        if len(feats) < page:
            break
    return np.asarray(points, dtype=np.float64)


print("\n[traffic] downloading Caltrans AADT points...")
pts = fetch_aadt()
print(f"[traffic] usable AADT points: {len(pts)}  (AADT range {pts[:,2].min():.0f}-{pts[:,2].max():.0f})")

traffic = np.zeros((nlat, nlon), dtype=np.float64)
lat_i = np.round((pts[:, 1] - glat[0]) / lat_res).astype(int)
lon_i = np.round((pts[:, 0] - glon[0]) / lon_res).astype(int)
inside = (lat_i >= 0) & (lat_i < nlat) & (lon_i >= 0) & (lon_i < nlon)
for li, oi, a in zip(lat_i[inside], lon_i[inside], pts[inside, 2]):
    if a > traffic[li, oi]:
        traffic[li, oi] = a  # busiest highway touching this cell
print(f"[traffic] cells with traffic: {int((traffic > 0).sum())} of {nlat*nlon}")

# log1p compresses the heavy AADT tail; smooth to spread influence to nearby cells.
traffic = np.log1p(traffic)
traffic = gaussian_filter(traffic, sigma=2.5)


# ---------------------------------------------------------------------------
# 2. ROAD DENSITY — read GRIP4 ESRI-ASCII grid and regrid (nearest) to 1km.
# ---------------------------------------------------------------------------
def read_asc(path):
    header = {}
    with open(path) as fh:
        for _ in range(6):
            key, val = fh.readline().split()
            header[key.lower()] = float(val)
        data = np.loadtxt(fh)
    ncols, nrows = int(header["ncols"]), int(header["nrows"])
    xll, yll, cs = header["xllcorner"], header["yllcorner"], header["cellsize"]
    nod = header["nodata_value"]
    lons = xll + (np.arange(ncols) + 0.5) * cs
    lats = yll + (np.arange(nrows) + 0.5) * cs         # south -> north
    data = data[::-1, :].astype(np.float64)            # flip rows to match ascending lat
    data[data == nod] = np.nan
    return lats, lons, data


print("\n[road] reading GRIP4 total road density...")
rlat, rlon, road = read_asc(grip_asc)
lat_idx = np.clip(np.round((glat - rlat[0]) / (rlat[1] - rlat[0])).astype(int), 0, len(rlat) - 1)
lon_idx = np.clip(np.round((glon - rlon[0]) / (rlon[1] - rlon[0])).astype(int), 0, len(rlon) - 1)
road_1km = np.nan_to_num(road[np.ix_(lat_idx, lon_idx)], nan=0.0)
print(f"[road] regridded to {road_1km.shape}; density range {road_1km.min():.0f}-{road_1km.max():.0f} m/km^2")


# ---------------------------------------------------------------------------
# 3. Save both, aligned to the master grid.
# ---------------------------------------------------------------------------
def save_nc(path, varname, data, units):
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("lat", nlat)
        ds.createDimension("lon", nlon)
        ds.createVariable("lat", "f8", ("lat",))[:] = glat
        ds.createVariable("lon", "f8", ("lon",))[:] = glon
        v = ds.createVariable(varname, "f4", ("lat", "lon"), zlib=True)
        v[:] = data.astype(np.float32)
        v.units = units


save_nc(processed / "traffic_aadt_1x1km.nc", "traffic_aadt", traffic, "log1p(AADT) smoothed")
save_nc(processed / "grip_road_density_1x1km.nc", "road_density", road_1km, "m/km2")
print("\n✅ Saved traffic_aadt_1x1km.nc and grip_road_density_1x1km.nc")
