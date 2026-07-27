import cdsapi
from pathlib import Path

# Set up the path
BASE_DIR = Path(__file__).resolve().parent.parent
out_dir = BASE_DIR / 'data' / 'raw' / 'ecmwf'
out_dir.mkdir(parents=True, exist_ok=True)

c = cdsapi.Client()

# List of the exact year/month combos for your TEMPO V03 data
months_to_download = [
    ('2023', '08'), ('2023', '09'), ('2023', '10'), ('2023', '11'), ('2023', '12'),
    ('2024', '01'), ('2024', '02'), ('2024', '03'), ('2024', '04'), ('2024', '05'),
    ('2024', '06'), ('2024', '07'), ('2024', '08'), ('2024', '09')
]

print("Starting monthly ERA5 downloads for California...")

for year, month in months_to_download:
    output_path = out_dir / f'era5_california_{year}_{month}.nc'
    
    # Skip if we already downloaded this month
    if output_path.exists():
        print(f"Skipping {year}-{month}, file already exists.")
        continue
        
    print(f"Requesting data for {year}-{month}...")
    
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '10m_u_component_of_wind', '10m_v_component_of_wind', 
                    '2m_temperature', 'boundary_layer_height', 'surface_pressure'
                ],
                'year': year,
                'month': month,
                # 31 days (Copernicus automatically ignores day 31 for shorter months)
                'day': [f"{i:02d}" for i in range(1, 32)],
                'time': [f"{i:02d}:00" for i in range(24)],
                'area': [42.5, -124.5, 32.5, -114.0],
            },
            str(output_path)
        )
        print(f"Successfully saved {year}-{month}")
    except Exception as e:
        print(f"Failed to download {year}-{month}. Error: {e}")

print("All requested months processed!")