import os
from pathlib import Path

try:
  import cdsapi
except ImportError:
  raise ImportError(
      "Please install cdsapi first by running: pip install cdsapi"
  )

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "era5_monthly_chunks"
RAW_DIR.mkdir(parents=True, exist_ok=True)

print(
    "🌧️ STARTING PATH B: COPERNICUS ERA5 MONTH-BY-MONTH CHUNKED DOWNLOAD 🌧️"
)
print("=" * 75)
print(
    "📦 Requesting 4 Chemical Weather Drivers in monthly chunks to bypass CDS"
    " limits..."
)

c = cdsapi.Client()

# ==============================================================================
# 2. MONTH-BY-MONTH DOWNLOAD LOOP (BYPASSES STRICT NETCDF COST LIMITS!)
# ==============================================================================
years = ["2023", "2024"]
months = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
]

for year in years:
  for month in months:
    out_file = RAW_DIR / f"era5_california_missing_vars_{year}_{month}.nc"

    # Skip if we already downloaded this month in a previous run!
    if out_file.exists() and out_file.stat().st_size > 10000:
      print(f"⏩ Skipping {year}-{month} (already exists on disk)...")
      continue

    print(f"\n⏳ Submitting request for {year}-{month} -> {out_file.name}...")

    request_params = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": [
            "surface_solar_radiation_downwards",  # ssrd (UV / Photolysis proxy)
            "2m_dewpoint_temperature",  # d2m  (Relative humidity / Nitrates)
            "total_cloud_cover",  # tcc  (Radiation blocker)
            "total_precipitation",  # tp   (Atmospheric rain scrubber)
        ],
        "year": [year],
        "month": [month],  # <--- SINGLE MONTH CHUNK!
        "day": [
            "01",
            "02",
            "03",
            "04",
            "05",
            "06",
            "07",
            "08",
            "09",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
            "22",
            "23",
            "24",
            "25",
            "26",
            "27",
            "28",
            "29",
            "30",
            "31",
        ],
        "time": [
            "00:00",
            "01:00",
            "02:00",
            "03:00",
            "04:00",
            "05:00",
            "06:00",
            "07:00",
            "08:00",
            "09:00",
            "10:00",
            "11:00",
            "12:00",
            "13:00",
            "14:00",
            "15:00",
            "16:00",
            "17:00",
            "18:00",
            "19:00",
            "20:00",
            "21:00",
            "22:00",
            "23:00",
        ],
        "area": [42.5, -124.5, 32.0, -113.5],
    }

    try:
      c.retrieve("reanalysis-era5-single-levels", request_params, str(out_file))
      print(f"✅ {year}-{month} downloaded successfully!")
    except Exception as e:
      print(f"❌ {year}-{month} download failed: {e}")

print("\n" + "=" * 75)
print("🎯 PATH B MONTHLY CHUNKED DOWNLOADS FINISHED!")