"""
ConvLSTM Training Script for Temporal Air Quality Forecasting
LA Basin Digital Twin - Hourly, 1km Resolution

Features:
- ConvLSTM model for temporal sequence forecasting
- Sequence-based data loading (T=4 → predict t+1)
- LA Basin spatial crop maintenance
- Multi-core CPU optimization (no CUDA)
- 24-hour diurnal validation with peak-hour analysis
"""

import os
import sys
from pathlib import Path

# Ensure scripts directory is in path for local imports
sys.path.insert(0, str(Path(__file__).parent))

import time
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import custom modules
from convlstm_architecture import ConvLSTMPredictor, ConvLSTMSimple
from tempo_convlstm_dataset import TEMPOConvLSTMDataset


# ==============================================================================
# 1. MULTI-CORE CPU OPTIMIZATION & DEVICE SETUP
# ==============================================================================
def setup_device_and_threads(num_threads=None):
    """Configure PyTorch for multi-core CPU execution without CUDA."""
    if num_threads is None:
        num_threads = min(8, os.cpu_count() or 4)
    
    device = torch.device('cpu')
    torch.set_num_threads(num_threads)
    
    # Optional: Set environment variables for additional optimization
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    
    print(f'🔥 PYTORCH MULTI-CORE CPU OPTIMIZATION')
    print(f'   Device: {device}')
    print(f'   Threads: {torch.get_num_threads()}')
    print(f'   CPU Count: {os.cpu_count()}')
    
    return device


# ==============================================================================
# 2. LA BASIN SPATIAL CROP CALCULATION
# ==============================================================================
def get_la_basin_crop(era5_file):
    """
    Dynamically calculate LA Basin spatial crop indices.
    
    LA Basin bounds (extended):
        Latitude: 33.30°N - 34.60°N
        Longitude: -119.00°W - -117.20°W
    
    Returns:
        (r_min, r_max, c_min, c_max): Row and column indices for cropping
    """
    print('⏳ Calculating LA Basin Spatial Crop Indices...')
    
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
    
    crop_height = r_max - r_min + 1
    crop_width = c_max - c_min + 1
    
    print(f'🎯 LA Basin Crop Matrix:')
    print(f'   Rows: [{r_min:3d}:{r_max:3d}] (height={crop_height} pixels)')
    print(f'   Cols: [{c_min:3d}:{c_max:3d}] (width={crop_width} pixels)')
    
    return r_min, r_max, c_min, c_max


# ==============================================================================
# 3. TRAINING LOOP WITH TEMPORAL SEQUENCES
# ==============================================================================
def train_convlstm_model(
    model, device, train_loader, criterion, optimizer,
    epochs=5, log_interval=10, max_batches_per_epoch=None
):
    """
    Training loop for ConvLSTM model.
    
    Args:
        model: ConvLSTM model
        device: torch device (cpu/cuda)
        train_loader: DataLoader for training
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of training epochs
        log_interval: Log interval (batches)
        max_batches_per_epoch: Cap on batches per epoch (for quick testing)
    
    Returns:
        losses: List of average losses per log interval
    """
    print('=' * 70)
    print('🚀 TRAINING CONVLSTM DIGITAL TWIN (LA BASIN CORRIDOR)')
    print('=' * 70)
    
    model.train()
    losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0
        
        for batch_idx, (x_seq, y_target) in enumerate(train_loader):
            if max_batches_per_epoch and batch_idx >= max_batches_per_epoch:
                print(f'🏁 Reached target benchmark step ({max_batches_per_epoch} batches).')
                break
            
            # Move to device (CPU in our case, but keep general)
            # x_seq: [Batch, Time=4, Channels=5, Height, Width]
            # y_target: [Batch, 1, Height, Width]
            x_seq = x_seq.to(device)
            y_target = y_target.to(device)
            
            # Apply LA Basin crop for faster computation on CPU
            x_seq = x_seq[:, :, :, r_min:r_max+1, c_min:c_max+1]
            y_target = y_target[:, :, r_min:r_max+1, c_min:c_max+1]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(x_seq)  # [Batch, 1, Height, Width]
            
            # Compute loss
            loss = criterion(predictions, y_target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            
            # Log progress
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                elapsed = time.time() - start_time
                print(
                    f'   Epoch [{epoch + 1}/{epochs}] | Batch [{batch_idx + 1:03d}] | '
                    f'Loss: {avg_loss:.6f} | Time: {elapsed:.1f}s'
                )
                losses.append(avg_loss)
                running_loss = 0.0
    
    total_time = time.time() - start_time
    print('=' * 70)
    print(f'🎉 CONVLSTM TRAINING COMPLETED IN {total_time:.1f} SECONDS!')
    print('=' * 70)
    
    return losses


# ==============================================================================
# 4. 24-HOUR DIURNAL VALIDATION WITH PEAK-HOUR ANALYSIS
# ==============================================================================
def generate_24hour_diurnal_validation(
    model, device, dataset, r_min, r_max, c_min, c_max,
    num_validation_sequences=24, output_dir=None
):
    """
    Generate 24-hour diurnal validation comparing predicted vs actual NO2.
    
    Creates a line chart showing:
    - 24-hour cycle of actual NO2
    - Predicted NO2 from ConvLSTM
    - Captures rush-hour peaks (morning and evening)
    
    Args:
        model: Trained ConvLSTM model
        device: torch device
        dataset: Dataset object (to access TEMPO NO2 data)
        r_min, r_max, c_min, c_max: LA Basin crop indices
        num_validation_sequences: Number of sequences to validate (hourly)
        output_dir: Directory to save visualization
    
    Returns:
        actual_no2_24h: Actual NO2 values (24,)
        predicted_no2_24h: Predicted NO2 values (24,)
    """
    print('\n🎨 Generating 24-Hour Diurnal Validation Profile...')
    print('   Scanning for representative urban plume station...')
    
    model.eval()
    
    actual_no2_24h = np.zeros(num_validation_sequences, dtype=np.float32)
    predicted_no2_24h = np.zeros(num_validation_sequences, dtype=np.float32)
    
    # Identify a representative urban grid cell (high NO2 signal)
    # Typically center-left of crop (Downtown LA vicinity)
    urban_row_offset = (r_max - r_min) // 3  # 1/3 from top
    urban_col_offset = (c_max - c_min) // 4  # 1/4 from left
    urban_row = urban_row_offset
    urban_col = urban_col_offset
    
    print(f'   Urban station reference: Row {urban_row}, Col {urban_col}')
    
    with torch.no_grad():
        # Sample sequences across validation set
        step_size = max(1, len(dataset) // num_validation_sequences)
        sample_indices = list(range(0, len(dataset), step_size))[:num_validation_sequences]
        
        for hour_idx, sample_idx in enumerate(sample_indices):
            try:
                x_seq, y_target = dataset[sample_idx]
                
                # Crop to LA Basin
                x_seq = x_seq[:, :, r_min:r_max+1, c_min:c_max+1]
                y_target = y_target[:, r_min:r_max+1, c_min:c_max+1]
                
                # Add batch dimension and move to device
                x_seq_batch = x_seq.unsqueeze(0).to(device)  # [1, T=4, C=5, H, W]
                
                # Model prediction
                y_pred = model(x_seq_batch).cpu().numpy()  # [1, 1, H, W]
                
                # Extract urban grid cell value (now within cropped coordinates)
                actual_val = y_target[0, urban_row, urban_col].item()
                pred_val = y_pred[0, 0, urban_row, urban_col].item()
                
                actual_no2_24h[hour_idx] = actual_val
                predicted_no2_24h[hour_idx] = pred_val
                
            except Exception as e:
                print(f'   ⚠️  Error at sample {sample_idx}: {e}')
                continue
    
    # Identify rush-hour peaks
    actual_peak_hour = np.argmax(actual_no2_24h)
    actual_peak_value = actual_no2_24h[actual_peak_hour]
    pred_peak_hour = np.argmax(predicted_no2_24h)
    pred_peak_value = predicted_no2_24h[pred_peak_hour]
    
    print(f'\n✅ Diurnal Validation Complete:')
    print(f'   Actual Peak: Hour {actual_peak_hour:2d} (NO₂={actual_peak_value:.4f})')
    print(f'   Pred Peak:   Hour {pred_peak_hour:2d} (NO₂={pred_peak_value:.4f})')
    
    # Create 24-hour line chart
    fig, ax = plt.subplots(figsize=(14, 6))
    
    hours = np.arange(num_validation_sequences)
    
    ax.plot(
        hours, actual_no2_24h,
        'o-', linewidth=2.5, markersize=6,
        label='TEMPO Ground Truth', color='#1f77b4', alpha=0.8
    )
    ax.plot(
        hours, predicted_no2_24h,
        's--', linewidth=2.5, markersize=6,
        label='ConvLSTM Prediction', color='#ff7f0e', alpha=0.8
    )
    
    # Highlight rush-hour peaks
    ax.scatter(
        [actual_peak_hour], [actual_peak_value],
        s=200, color='#1f77b4', marker='*', zorder=5,
        label=f'Actual Peak (Hour {actual_peak_hour})', edgecolor='black', linewidth=1
    )
    ax.scatter(
        [pred_peak_hour], [pred_peak_value],
        s=200, color='#ff7f0e', marker='*', zorder=5,
        label=f'Predicted Peak (Hour {pred_peak_hour})', edgecolor='black', linewidth=1
    )
    
    # Shade typical morning (6-9am) and evening (4-7pm) rush hours
    ax.axvspan(6, 9, alpha=0.15, color='red', label='Morning Rush')
    ax.axvspan(16, 19, alpha=0.15, color='darkred', label='Evening Rush')
    
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('NO₂ VCD (Normalized)', fontsize=12, fontweight='bold')
    ax.set_title(
        'ConvLSTM Digital Twin: 24-Hour Diurnal NO₂ Profile (Urban Station)',
        fontsize=14, fontweight='bold'
    )
    ax.set_xticks(np.arange(0, num_validation_sequences, 3))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    
    if output_dir:
        val_path = Path(output_dir) / 'convlstm_24h_diurnal_validation.png'
        plt.savefig(val_path, dpi=300, bbox_inches='tight')
        print(f'\n💾 Diurnal validation chart saved to:\n   {val_path}')
    
    plt.show()
    
    return actual_no2_24h, predicted_no2_24h


# ==============================================================================
# 5. SPATIAL VALIDATION MAP (SINGLE SNAPSHOT)
# ==============================================================================
def generate_spatial_validation_map(
    model, device, dataset, r_min, r_max, c_min, c_max,
    output_dir=None, num_samples_to_scan=15
):
    """
    Generate spatial validation maps comparing predicted vs actual NO2.
    
    Scans validation set to find a clear daylight urban plume.
    
    Args:
        model: Trained ConvLSTM model
        device: torch device
        dataset: Dataset object
        r_min, r_max, c_min, c_max: LA Basin crop indices
        output_dir: Directory to save visualization
        num_samples_to_scan: Number of samples to scan for best plume
    
    Returns:
        y_true, y_pred: Ground truth and predictions
    """
    print('\n🎨 Generating Spatial Validation Map...')
    print('   Searching for high-signal daylight urban plume...')
    
    model.eval()
    
    best_x, best_y = None, None
    max_signal = -1.0
    
    with torch.no_grad():
        # Scan samples across dataset
        step_size = max(1, len(dataset) // num_samples_to_scan)
        sample_indices = list(range(0, len(dataset), step_size))[:num_samples_to_scan]
        
        for sample_idx in sample_indices:
            try:
                x_seq, y_target = dataset[sample_idx]
                
                # Crop to LA Basin
                x_seq_cropped = x_seq[:, :, r_min:r_max+1, c_min:c_max+1]
                y_target_cropped = y_target[:, r_min:r_max+1, c_min:c_max+1]
                
                signal_strength = y_target_cropped.max().item()
                
                if signal_strength > max_signal:
                    max_signal = signal_strength
                    best_x = x_seq_cropped.unsqueeze(0).to(device)  # [1, T=4, C=5, H, W]
                    best_y = y_target_cropped
            except Exception as e:
                continue
        
        if best_x is None:
            print('   ⚠️  Could not find valid validation samples')
            return None, None
        
        y_pred = model(best_x).cpu().numpy()
        y_true = best_y.numpy()
    
    print(f'✅ Selected daylight validation snapshot with peak NO₂: {max_signal:.6f}')
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)
    fig.suptitle(
        'ConvLSTM Digital Twin: TEMPO Ground Truth vs. Prediction',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    # Dynamic scaling based on actual signal
    vmin_val = max(0, np.percentile(y_true[0], 5))
    vmax_val = np.percentile(y_true[0], 95)
    
    im0 = axes[0].imshow(y_true[0], cmap='jet', origin='lower', vmin=vmin_val, vmax=vmax_val)
    axes[0].set_title('TEMPO Satellite (Ground Truth)', fontsize=12, fontweight='bold')
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label('NO₂ VCD (Normalized)', fontsize=10)
    
    im1 = axes[1].imshow(y_pred[0, 0], cmap='jet', origin='lower', vmin=vmin_val, vmax=vmax_val)
    axes[1].set_title('ConvLSTM Prediction', fontsize=12, fontweight='bold', color='darkblue')
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label('Predicted NO₂ (Normalized)', fontsize=10)
    
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlabel('Grid Column (X)')
        ax.set_ylabel('Grid Row (Y)')
    
    plt.tight_layout()
    
    if output_dir:
        map_path = Path(output_dir) / 'convlstm_spatial_validation.png'
        plt.savefig(map_path, dpi=300, bbox_inches='tight')
        print(f'\n💾 Spatial validation map saved to:\n   {map_path}')
    
    plt.show()
    
    return y_true, y_pred


# ==============================================================================
# MAIN TRAINING EXECUTION
# ==============================================================================
if __name__ == '__main__':
    print('╔' + '=' * 68 + '╗')
    print('║' + ' CONVLSTM DIGITAL TWIN FOR LA BASIN AIR QUALITY FORECASTING '.center(68) + '║')
    print('╚' + '=' * 68 + '╝')
    
    # Setup paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    era5_file = BASE_DIR / 'data' / 'processed' / 'era5_california_1x1km_master.nc'
    tempo_dir = BASE_DIR / 'data' / 'processed' / 'tempo_monthly'
    output_dir = BASE_DIR / 'data' / 'processed'
    model_dir = BASE_DIR / 'models'
    model_dir.mkdir(exist_ok=True)
    
    # Setup device and threading
    device = setup_device_and_threads(num_threads=8)
    print()
    
    # Get LA Basin crop indices
    r_min, r_max, c_min, c_max = get_la_basin_crop(era5_file)
    print()
    
    # Initialize dataset and dataloader
    print('⏳ Initializing ConvLSTM Dataset with Temporal Sequences...')
    dataset = TEMPOConvLSTMDataset(
        era5_file, tempo_dir,
        sequence_length=4,
        normalize=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,        # Reduced from 4 to avoid memory issues
        shuffle=True,
        num_workers=1,       # Reduced from 2 to reduce worker overhead
        pin_memory=False,
        drop_last=True
    )
    print()
    
    # Initialize model
    print('🏗️  Building ConvLSTM Architecture...')
    # Choose between ConvLSTMPredictor (with decoder) or ConvLSTMSimple (lightweight)
    model = ConvLSTMSimple(
        in_channels=5,
        hidden_channels=32,
        num_layers=3,
        output_channels=1
    ).to(device)
    
    print(f'   Model Type: ConvLSTMSimple')
    print(f'   Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print()
    
    # Setup optimizer and loss
    criterion = nn.SmoothL1Loss()  # Robust to spikes
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Training configuration
    EPOCHS = 2
    LOG_INTERVAL = 10
    MAX_BATCHES = 300  # Increased from 150 (batch_size reduced 4→2, so double batches for same samples)
    
    # Train model
    losses = train_convlstm_model(
        model, device, dataloader, criterion, optimizer,
        epochs=EPOCHS,
        log_interval=LOG_INTERVAL,
        max_batches_per_epoch=MAX_BATCHES
    )
    print()
    
    # Save model
    model_path = model_dir / 'convlstm_digital_twin_la_basin.pth'
    torch.save(model.state_dict(), model_path)
    print(f'💾 Trained ConvLSTM weights saved to:\n   {model_path}')
    print()
    
    # Generate validation visualizations
    # Spatial map
    y_true_spatial, y_pred_spatial = generate_spatial_validation_map(
        model, device, dataset, r_min, r_max, c_min, c_max,
        output_dir=output_dir, num_samples_to_scan=15
    )
    
    # 24-hour diurnal profile
    actual_24h, pred_24h = generate_24hour_diurnal_validation(
        model, device, dataset, r_min, r_max, c_min, c_max,
        num_validation_sequences=24,
        output_dir=output_dir
    )
    
    # Print summary statistics
    print('\n' + '=' * 70)
    print('📊 TRAINING SUMMARY')
    print('=' * 70)
    print(f'Architecture:      ConvLSTMSimple (3 layers, 32-128 channels)')
    print(f'Input Sequence:    T=4 timesteps of 5 meteorological channels')
    print(f'Output:            1-channel NO2 prediction at t+1')
    print(f'LA Basin Crop:     [{r_min}:{r_max}, {c_min}:{c_max}]')
    print(f'Device:            {device} (multi-core optimized)')
    print(f'Training Batches:  {MAX_BATCHES}')
    print(f'Loss Function:     SmoothL1 (robust to extremes)')
    print(f'Optimizer:         AdamW (lr=1e-3, weight_decay=1e-4)')
    
    if len(actual_24h) > 0:
        mae_24h = np.mean(np.abs(actual_24h - pred_24h))
        rmse_24h = np.sqrt(np.mean((actual_24h - pred_24h) ** 2))
        print(f'\n24-Hour Validation Metrics:')
        print(f'  MAE:  {mae_24h:.6f}')
        print(f'  RMSE: {rmse_24h:.6f}')
    
    print('=' * 70)
    print('✨ CONVLSTM TRAINING COMPLETE!')
    print('=' * 70)
