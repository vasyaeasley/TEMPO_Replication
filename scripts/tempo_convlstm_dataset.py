"""
Updated PyTorch Dataset for ConvLSTM temporal sequences.

Loads sequences of T=4 consecutive timesteps + target t+1.
Input shape: [Batch, Time=4, Channels=5, Height, Width]
Target shape: [Batch, 1, Height, Width]
"""
import time
from pathlib import Path
import netCDF4 as nc
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TEMPOConvLSTMDataset(Dataset):
    """
    Temporal ConvLSTM Dataset: Load sequences for next-timestep forecasting.
    
    For each index, returns:
        - Input: [t-3, t-2, t-1, t] with shape [T=4, Channels=5, H, W]
        - Target: t+1 with shape [Channels=1, H, W]
    """

    def __init__(self, era5_master_path, tempo_dir, sequence_length=4, normalize=True):
        """
        Args:
            era5_master_path: Path to master ERA5 NetCDF file
            tempo_dir: Directory containing monthly TEMPO NO2 files
            sequence_length: Number of timesteps in sequence (default T=4)
            normalize: Whether to apply normalization
        """
        self.era5_ds = None
        self.tempo_ds = None
        self.current_tempo_idx = -1

        self.era5_path = Path(era5_master_path)
        self.tempo_files = sorted(list(Path(tempo_dir).glob('*.nc')))
        self.sequence_length = sequence_length
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

        # Valid indices: must have (sequence_length-1) prior timesteps and 1 future timestep
        # So index 0 has no prior history, index (total_hours-1) has no future target
        # Valid range: [sequence_length - 1, total_hours - 1)
        self.valid_start = sequence_length - 1
        self.valid_end = self.total_hours - 1
        self.num_valid_samples = max(0, self.valid_end - self.valid_start)

        print(
            f'✅ ConvLSTM Dataset Initialized: {self.total_hours} total timestamps available.'
        )
        print(f'   Time Coordinate Key: [{self.time_key}]')
        print(f'   Variable Mapping:    {self.var_map}')
        print(f'   Grid Dimensions:     [{self.height} rows x {self.width} cols]')
        print(f'   Sequence Length:     {self.sequence_length}')
        print(
            f'   Valid Sequence Range: [{self.valid_start}, {self.valid_end}) '
            f'({self.num_valid_samples} valid sequences)'
        )

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
        """Return number of valid sequences (not total timesteps)."""
        return self.num_valid_samples

    def __getitem__(self, idx):
        """
        Load a sequence of T timesteps + target t+1.

        Args:
            idx: Index in range [0, num_valid_samples)

        Returns:
            x_sequence: Tensor [T, Channels=5, Height, Width] - sequence of T meteorological snapshots
            y_target: Tensor [1, Height, Width] - NO2 target at t+1
        """
        # Convert valid sequence index to absolute time index
        # idx=0 corresponds to target at valid_start+1
        target_time_idx = self.valid_start + idx + 1
        
        # Collect timestep indices for the sequence [t-3, t-2, t-1, t]
        sequence_indices = [
            target_time_idx - self.sequence_length,
            target_time_idx - self.sequence_length + 1,
            target_time_idx - self.sequence_length + 2,
            target_time_idx - self.sequence_length + 3,
        ]

        # Allocate memory for sequence: [T, Channels=5, H, W]
        x_sequence = np.zeros(
            (self.sequence_length, 5, self.height, self.width),
            dtype=np.float32
        )

        # Load each timestep in the sequence
        for seq_pos, time_idx in enumerate(sequence_indices):
            # Map global hour index to the correct monthly TEMPO file (~730 hours/month)
            tempo_file_idx = min(time_idx // 730, len(self.tempo_files) - 1)
            
            # Safely open file handles within this worker's memory space
            self._init_worker_files(tempo_file_idx)

            # Extract the 5 meteorological covariates using our universal map
            t2m = self.era5_ds.variables[self.var_map['t2m']][time_idx, :, :]
            u10 = self.era5_ds.variables[self.var_map['u10']][time_idx, :, :]
            v10 = self.era5_ds.variables[self.var_map['v10']][time_idx, :, :]
            sp = self.era5_ds.variables[self.var_map['sp']][time_idx, :, :]
            blh = self.era5_ds.variables[self.var_map['blh']][time_idx, :, :]

            # Stack into multi-channel array: Shape [5, H, W]
            weather_snapshot = np.stack(
                [t2m, u10, v10, sp, blh],
                axis=0
            ).astype(np.float32)

            # Apply normalization if requested
            if self.normalize:
                weather_snapshot[0] = (weather_snapshot[0] - 285.0) / 15.0  # Temperature
                weather_snapshot[1] = weather_snapshot[1] / 10.0  # U-Wind
                weather_snapshot[2] = weather_snapshot[2] / 10.0  # V-Wind
                weather_snapshot[3] = (weather_snapshot[3] - 95000.0) / 5000.0  # Pressure
                weather_snapshot[4] = weather_snapshot[4] / 1000.0  # Boundary Layer Height

            x_sequence[seq_pos] = weather_snapshot

        # Load target TEMPO NO2 at t+1
        tempo_file_idx = min(target_time_idx // 730, len(self.tempo_files) - 1)
        self._init_worker_files(tempo_file_idx)
        
        local_time_idx = target_time_idx % 730
        safe_tempo_idx = local_time_idx % self.tempo_ds.variables['NO2_column'].shape[0]
        
        y_no2 = self.tempo_ds.variables['NO2_column'][safe_tempo_idx, :, :]
        y_no2 = np.expand_dims(y_no2, axis=0).astype(np.float32)

        if self.normalize:
            y_no2 = np.nan_to_num(y_no2, nan=0.0) / 1e16  # NO2 Scale

        return torch.from_numpy(x_sequence), torch.from_numpy(y_no2)

    def __del__(self):
        """Clean up file handles."""
        if getattr(self, 'era5_ds', None) is not None:
            self.era5_ds.close()
        if getattr(self, 'tempo_ds', None) is not None:
            self.tempo_ds.close()


# Keep the original dataset for backwards compatibility
class TEMPODigitalTwinDataset(Dataset):
    """Original single-timestep dataset (kept for reference/legacy code)."""

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

    print('🚀 Initializing ConvLSTM PyTorch Dataset...')
    dataset = TEMPOConvLSTMDataset(era5_file, tempo_dir, sequence_length=4, normalize=True)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
        drop_last=True,
    )

    print('\n⚡ Testing Temporal Batch Loading Speed...')
    start_time = time.time()
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        load_time = (time.time() - start_time) * 1000
        print(
            f'✅ Batch {batch_idx + 1} loaded in {load_time:.2f} ms!'
        )
        print(f'   Input shape (sequence):  {x_batch.shape}  (expected: [B, T=4, C=5, H, W])')
        print(f'   Target shape (t+1):      {y_batch.shape}  (expected: [B, 1, H, W])')
        
        if batch_idx >= 2:
            break

    print('\n✨ ConvLSTM dataset working correctly!')
