import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

os.environ['OMP_NUM_THREADS'] = '4'
import torch
torch.set_num_threads(4)

from scripts.convlstm_architecture import ConvLSTMSimple
from scripts.tempo_convlstm_dataset import TEMPOConvLSTMDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import netCDF4 as nc
import numpy as np

print('Step 1: Get LA Basin crop indices...', flush=True)
BASE_DIR = Path(__file__).resolve().parent
era5_file = BASE_DIR / 'data' / 'processed' / 'era5_california_1x1km_master.nc'
tempo_dir = BASE_DIR / 'data' / 'processed' / 'tempo_monthly'

# Get crop indices
with nc.Dataset(era5_file, 'r') as ds:
    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    if lats.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lons, lats)
    else:
        lat_grid, lon_grid = lats, lons

lat_mask = (lat_grid >= 33.30) & (lat_grid <= 34.60)
lon_mask = (lon_grid >= -119.00) & (lon_grid <= -117.20)
rows, cols = np.where(lat_mask & lon_mask)
r_min, r_max = rows.min(), rows.max()
c_min, c_max = cols.min(), cols.max()

print(f'LA Basin crop: [{r_min}:{r_max}, {c_min}:{c_max}] = {r_max-r_min+1}×{c_max-c_min+1} pixels', flush=True)

print('\nStep 2: Load dataset...', flush=True)
dataset = TEMPOConvLSTMDataset(era5_file, tempo_dir, sequence_length=4, normalize=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False, drop_last=True)

device = torch.device('cpu')
# Make model match crop size
crop_h = r_max - r_min + 1
crop_w = c_max - c_min + 1
print(f'Model will train on {crop_h}×{crop_w} LA Basin crop', flush=True)

model = ConvLSTMSimple(in_channels=5, hidden_channels=32, num_layers=3, output_channels=1).to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

print('\nStep 3: Test 3 batches with LA Basin crop...\n', flush=True)

for batch_idx, (x_seq, y_target) in enumerate(dataloader):
    if batch_idx >= 3:
        break
    
    print(f'Batch {batch_idx + 1}: Full shape x={x_seq.shape}, y={y_target.shape}', flush=True)
    
    # CROP to LA Basin
    x_seq = x_seq[:, :, :, r_min:r_max+1, c_min:c_max+1]
    y_target = y_target[:, :, r_min:r_max+1, c_min:c_max+1]
    
    print(f'Batch {batch_idx + 1}: Cropped x={x_seq.shape}, y={y_target.shape}', flush=True)
    
    x_seq = x_seq.to(device)
    y_target = y_target.to(device)
    
    print(f'Batch {batch_idx + 1}: Forward...', flush=True)
    optimizer.zero_grad()
    predictions = model(x_seq)
    loss = criterion(predictions, y_target)
    
    print(f'Batch {batch_idx + 1}: Backward...', flush=True)
    loss.backward()
    optimizer.step()
    
    print(f'✅ Batch {batch_idx + 1} complete! Loss: {loss.item():.6f}\n', flush=True)

print('Test complete! If you saw 3 batches complete quickly, the issue was the full grid size.')
