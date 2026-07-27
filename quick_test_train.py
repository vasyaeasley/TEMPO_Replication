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

print('Loading dataset...', flush=True)
BASE_DIR = Path(__file__).resolve().parent
era5_file = BASE_DIR / 'data' / 'processed' / 'era5_california_1x1km_master.nc'
tempo_dir = BASE_DIR / 'data' / 'processed' / 'tempo_monthly'

dataset = TEMPOConvLSTMDataset(era5_file, tempo_dir, sequence_length=4, normalize=True)

# KEY FIX: Reduce batch size and num_workers
print('Creating dataloader...', flush=True)
dataloader = DataLoader(
    dataset,
    batch_size=1,          # REDUCED from 4
    shuffle=True,
    num_workers=0,         # DISABLED workers to test
    pin_memory=False,
    drop_last=True
)

device = torch.device('cpu')
model = ConvLSTMSimple(in_channels=5, hidden_channels=32, num_layers=3, output_channels=1).to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

print('Starting training - loading first batch...', flush=True)

running_loss = 0.0
for batch_idx, (x_seq, y_target) in enumerate(dataloader):
    if batch_idx >= 5:  # Just 5 batches for speed test
        break
    
    print(f'Batch {batch_idx + 1}: Loading...', flush=True)
    x_seq = x_seq.to(device)
    y_target = y_target.to(device)
    
    print(f'Batch {batch_idx + 1}: Forward pass...', flush=True)
    optimizer.zero_grad()
    predictions = model(x_seq)
    loss = criterion(predictions, y_target)
    
    print(f'Batch {batch_idx + 1}: Backward pass...', flush=True)
    loss.backward()
    optimizer.step()
    
    print(f'✅ Batch [{batch_idx + 1}/5] | Loss: {loss.item():.6f}', flush=True)

print('Quick test complete!')
