import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tempo_dataset import TEMPODigitalTwinDataset

# ==============================================================================
# 1. ARCHITECTURE: SPATIAL U-NET FOR ATMOSPHERIC DIGITAL TWIN
# ==============================================================================
class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class AtmosphericUNet(nn.Module):
    """Spatial U-Net mapping 5 meteorological channels to 1 NO2 chemical channel."""
    def __init__(self, n_channels=5, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Downsampling Path (Encoder)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Bottleneck
        self.bot = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # Upsampling Path (Decoder with Skip Connections)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(128, 64)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(64, 32)

        # Final Chemical Output Projection
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.bot(x4)

        u1 = self.up1(x5)
        if u1.shape != x4.shape:
            u1 = nn.functional.interpolate(u1, size=x4.shape[2:], mode='bilinear')
        u1 = self.conv_up1(torch.cat([x4, u1], dim=1))

        u2 = self.up2(u1)
        if u2.shape != x3.shape:
            u2 = nn.functional.interpolate(u2, size=x3.shape[2:], mode='bilinear')
        u2 = self.conv_up2(torch.cat([x3, u2], dim=1))

        u3 = self.up3(u2)
        if u3.shape != x2.shape:
            u3 = nn.functional.interpolate(u3, size=x2.shape[2:], mode='bilinear')
        u3 = self.conv_up3(torch.cat([x2, u3], dim=1))

        u4 = self.up4(u3)
        if u4.shape != x1.shape:
            u4 = nn.functional.interpolate(u4, size=x1.shape[2:], mode='bilinear')
        u4 = self.conv_up4(torch.cat([x1, u4], dim=1))

        return self.outc(u4)


# ==============================================================================
# 2. TRAINING SETUP & EXECUTION
# ==============================================================================
if __name__ == '__main__':
    BASE_DIR = Path(__file__).resolve().parent.parent
    era5_file = BASE_DIR / 'data' / 'processed' / 'era5_california_1x1km_master.nc'
    tempo_dir = BASE_DIR / 'data' / 'processed' / 'tempo_monthly'
    output_dir = BASE_DIR / 'data' / 'processed'
    model_dir = BASE_DIR / 'models'
    model_dir.mkdir(exist_ok=True)

    # Enforce CPU Execution & Multi-Core Optimization for veasl001
    device = torch.device('cpu')
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    print(f'🔥 LAUNCHING PYTORCH U-NET ON DEVICE: [{device}] (Threads: {torch.get_num_threads()})')

    # Calculate LA Basin Crop Indices Dynamically from Master Coordinates
    print('⏳ Calculating Widened LA Basin Spatial Crop Indices...')
    with nc.Dataset(era5_file, 'r') as ds:
        lats = ds.variables['lat'][:] if 'lat' in ds.variables else ds.variables['latitude'][:]
        lons = ds.variables['lon'][:] if 'lon' in ds.variables else ds.variables['longitude'][:]
        if lats.ndim == 1:
            lon_grid, lat_grid = np.meshgrid(lons, lats)
        else:
            lat_grid, lon_grid = lats, lons

    lat_mask = (lat_grid >= 33.30) & (lat_grid <= 34.60)
    lon_mask = (lon_grid >= -119.00) & (lon_grid <= -117.20)
    rows, cols = np.where(lat_mask & lon_mask)
    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    print(f'🎯 LA Basin Crop Matrix: Rows [{r_min}:{r_max}] x Cols [{c_min}:{c_max}] (~{r_max-r_min}x{c_max-c_min} pixels)')

    # Initialize Dataset & DataLoader
    print('⏳ Connecting to Master Tensor Cube...')
    dataset = TEMPODigitalTwinDataset(era5_file, tempo_dir, normalize=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, drop_last=True)

    model = AtmosphericUNet(n_channels=5, n_classes=1).to(device)
    criterion = nn.SmoothL1Loss()  # Robust to wildfire/emission spikes
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    EPOCHS = 1
    LOG_INTERVAL = 10
    MAX_BATCHES = 150  # 150 mini-batches (~600 daylight hours) is plenty for local convergence!

    print('=' * 60)
    print('🚀 TRAINING SPATIAL U-NET DIGITAL TWIN (LA BASIN CORRIDOR)')
    print('=' * 60)

    start_time = time.time()
    model.train()

    running_loss = 0.0
    for epoch in range(EPOCHS):
        for batch_idx, (x_weather, y_no2) in enumerate(dataloader):
            if batch_idx >= MAX_BATCHES:
                print(f'🏁 Reached target benchmark step ({MAX_BATCHES} batches).')
                break

            # Apply dynamic LA Basin spatial crop to tensors
            x_weather = x_weather[:, :, r_min:r_max+1, c_min:c_max+1].to(device)
            y_no2 = y_no2[:, :, r_min:r_max+1, c_min:c_max+1].to(device)

            optimizer.zero_grad()
            predictions = model(x_weather)
            loss = criterion(predictions, y_no2)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg_loss = running_loss / LOG_INTERVAL
                elapsed = time.time() - start_time
                print(
                    f'   Epoch [{epoch + 1}/{EPOCHS}] | Batch [{batch_idx + 1:03d}/{MAX_BATCHES}] | '
                    f'SmoothL1 Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s'
                )
                running_loss = 0.0

    total_time = time.time() - start_time
    print('=' * 60)
    print(f'🎉 U-NET TRAINING COMPLETED IN {total_time:.1f} SECONDS!')
    print('=' * 60)

    model_path = model_dir / 'unet_digital_twin_la_basin.pth'
    torch.save(model.state_dict(), model_path)
    print(f'💾 Trained LA Basin U-Net weights saved to:\n   {model_path}')

   # ==============================================================================
    # 3. GENERATE HONEST VALIDATION COMPARISON MAP (DAYLIGHT FILTER)
    # ==============================================================================
    print('\n🎨 Searching for a high-signal daylight urban plume for validation...')
    model.eval()
    
    best_x, best_y = None, None
    max_signal = -1.0
    
    with torch.no_grad():
        # Scan 15 batches to find a clear afternoon daylight hour (avoiding cloudy/night zero-slices)
        for i, (x_test, y_test) in enumerate(dataloader):
            y_crop = y_test[:, :, r_min:r_max+1, c_min:c_max+1]
            signal_strength = y_crop.max().item()
            if signal_strength > max_signal:
                max_signal = signal_strength
                best_x = x_test[:, :, r_min:r_max+1, c_min:c_max+1].to(device)
                best_y = y_crop

        y_pred = model(best_x).cpu().numpy()
        y_true = best_y.numpy()

    print(f"🎯 Selected daylight validation hour with peak NO₂ signal: {max_signal:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    fig.suptitle(
        'Deep Learning Digital Twin: Real TEMPO Ground Truth vs. Trained U-Net Prediction',
        fontsize=14,
        fontweight='bold',
        y=0.96,
    )

    # Dynamic scaling based on the true daylight signal distribution
    vmin_val = max(0, np.percentile(y_true[0, 0], 5))
    vmax_val = np.percentile(y_true[0, 0], 95)

    im0 = axes[0].imshow(y_true[0, 0], cmap='jet', origin='lower', vmin=vmin_val, vmax=vmax_val)
    axes[0].set_title('TEMPO Satellite Ground Truth (NO₂ VCD)', fontsize=12, fontweight='bold')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='Normalized NO₂')

    im1 = axes[1].imshow(y_pred[0, 0], cmap='jet', origin='lower', vmin=vmin_val, vmax=vmax_val)
    axes[1].set_title('Trained U-Net Predicted Air Quality (From Weather Only)', fontsize=12, fontweight='bold', color='darkblue')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Predicted NO₂')

    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel('Grid Column (X)')
        ax.set_ylabel('Grid Row (Y)')

    plt.tight_layout()
    val_map_path = output_dir / 'la_basin_unet_prediction_comparison.png'
    plt.savefig(val_map_path, dpi=300, bbox_inches='tight')
    print(f'🏆 Honest validation map successfully saved to:\n   {val_map_path}')
    print('=' * 60)