import time
from pathlib import Path
import netCDF4 as nc
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TEMPODigitalTwinDataset(Dataset):

  def __init__(self, era5_master_path, tempo_dir, normalize=True):
    """PyTorch Dataset for pairing 1km ERA5 meteorology with TEMPO NO2

    satellite swaths.
    """
    self.era5_ds = None
    self.tempo_ds = None
    self.current_tempo_idx = -1

    self.era5_path = Path(era5_master_path)
    self.tempo_files = sorted(list(Path(tempo_dir).glob('*.nc')))
    self.normalize = normalize

    if not self.era5_path.exists() or not self.tempo_files:
      raise FileNotFoundError('Missing master ERA5 file or TEMPO monthly files!')

    # Read ONLY metadata in __init__ (prevents multi-worker deadlocks!)
    with nc.Dataset(self.era5_path, 'r') as ds:
      # Universal Coordinate Finder
      self.time_key = (
          'time'
          if 'time' in ds.variables
          else (
              'valid_time'
              if 'valid_time' in ds.variables
              else list(ds.dimensions.keys())[0]
          )
      )
      self.lat_key = 'lat' if 'lat' in ds.variables else 'latitude'
      self.lon_key = 'lon' if 'lon' in ds.variables else 'longitude'

      # Universal Variable Map: Automatically links ECMWF short codes to expected inputs
      expected_vars = {
          't2m': ['t2m', '2m_temperature', 'temp'],
          'u10': ['u10', '10m_u_wind', 'u_wind'],
          'v10': ['v10', '10m_v_wind', 'v_wind'],
          'sp': ['sp', 'surface_pressure', 'pressure'],
          'blh': ['blh', 'boundary_layer_height', 'pblh'],
      }
      self.var_map = {}
      for target, candidates in expected_vars.items():
        for cand in candidates:
          if cand in ds.variables:
            self.var_map[target] = cand
            break
        if target not in self.var_map:
          raise KeyError(
              f'Could not find a matching variable in NetCDF for: {target}'
          )

      self.total_hours = len(ds.variables[self.time_key])
      self.height = ds.variables[self.lat_key].shape[0]
      self.width = ds.variables[self.lon_key].shape[0]

    print(
        f'✅ Dataset Initialized: {self.total_hours} total timestamps available.'
    )
    print(f'   Time Coordinate Key: [{self.time_key}]')
    print(f'   Variable Mapping:    {self.var_map}')
    print(f'   Grid Dimensions:     [{self.height} rows x {self.width} cols]')

  def _init_worker_files(self, tempo_file_idx):
    """Lazily opens file handles inside the specific CPU worker process."""
    if self.era5_ds is None:
      self.era5_ds = nc.Dataset(self.era5_path, 'r')

    if self.current_tempo_idx != tempo_file_idx:
      if self.tempo_ds is not None:
        self.tempo_ds.close()
      self.tempo_ds = nc.Dataset(self.tempo_files[tempo_file_idx], 'r')
      self.current_tempo_idx = tempo_file_idx

  def __len__(self):
    return self.total_hours

  def __getitem__(self, idx):
    # Map global hour index to the correct monthly TEMPO file (~730 hours/month)
    tempo_file_idx = min(idx // 730, len(self.tempo_files) - 1)
    local_time_idx = idx % 730

    # 1. Safely open file handles within this worker's memory space
    self._init_worker_files(tempo_file_idx)

    # 2. Extract the 5 meteorological covariates using our universal map!
    t2m = self.era5_ds.variables[self.var_map['t2m']][idx, :, :]
    u10 = self.era5_ds.variables[self.var_map['u10']][idx, :, :]
    v10 = self.era5_ds.variables[self.var_map['v10']][idx, :, :]
    sp = self.era5_ds.variables[self.var_map['sp']][idx, :, :]
    blh = self.era5_ds.variables[self.var_map['blh']][idx, :, :]

    # 3. Stack into a multi-channel NumPy array: Shape [5, H, W]
    x_weather = np.stack([t2m, u10, v10, sp, blh], axis=0).astype(np.float32)

    # 4. Extract target TEMPO NO2 column: Shape [1, H, W]
    safe_tempo_idx = local_time_idx % self.tempo_ds.variables['NO2_column'].shape[0]
    y_no2 = self.tempo_ds.variables['NO2_column'][safe_tempo_idx, :, :]
    y_no2 = np.expand_dims(y_no2, axis=0).astype(np.float32)

    # 5. Fast on-the-fly normalization
    if self.normalize:
      x_weather[0] = (x_weather[0] - 285.0) / 15.0  # Temperature
      x_weather[1] = x_weather[1] / 10.0  # U-Wind
      x_weather[2] = x_weather[2] / 10.0  # V-Wind
      x_weather[3] = (x_weather[3] - 95000.0) / 5000.0  # Pressure
      x_weather[4] = x_weather[4] / 1000.0  # Boundary Layer Height
      y_no2 = np.nan_to_num(y_no2, nan=0.0) / 1e16  # NO2 Scale

    return torch.from_numpy(x_weather), torch.from_numpy(y_no2)

  def __del__(self):
    if getattr(self, 'era5_ds', None) is not None:
      self.era5_ds.close()
    if getattr(self, 'tempo_ds', None) is not None:
      self.tempo_ds.close()


if __name__ == '__main__':
  BASE_DIR = Path(__file__).resolve().parent.parent
  era5_file = BASE_DIR / 'data' / 'processed' / 'era5_california_1x1km_master.nc'
  tempo_dir = BASE_DIR / 'data' / 'processed' / 'tempo_monthly'

  print('🚀 Initializing PyTorch Dataset & GPU DataLoader...')
  dataset = TEMPODigitalTwinDataset(era5_file, tempo_dir, normalize=True)

  # Changed pin_memory=False to prevent warning when on CPU mode!
  dataloader = DataLoader(
      dataset,
      batch_size=4,
      shuffle=True,
      num_workers=2,
      pin_memory=False,
      drop_last=True,
  )

  print('\n⚡ Testing Batch Loading Speed...')
  start_time = time.time()
  for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
    load_time = (time.time() - start_time) * 1000
    print(f'✅ Batch {batch_idx + 1} loaded in {load_time:.2f} ms!')
    print(
        f'   Input Weather Tensor Shape:  {x_batch.shape} -> [Batch, Channels, Height, Width]'
    )
    print(f'   Target NO2 Tensor Shape:     {y_batch.shape}')
    break

  print(
      '\n🎯 Phase 3 Data Pipeline is 100% Operational and Ready for Training!'
  )