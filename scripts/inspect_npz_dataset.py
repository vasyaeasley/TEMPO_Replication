#!/usr/bin/env python3
"""
Master Dataset Structural & Metadata Inspection Tool
Path: /mnt/data3/veasl001/TEMPO_Replication/inspect_npz_dataset.py

Purpose:
    Performs deep inspection of 'epa_point_dataset_14months_20features.npz'
    to evaluate feature vector indexing, group grouping semantics (K-means 
    clusters vs station IDs vs temporal blocks), and missing column/coordinate 
    encodings prior to derivative ($\Delta NO_2$) and ST-GNN construction.
"""

import os
import sys

# ==============================================================================
# THREAD-SAFETY CONTROLS (MUST EXECUTE BEFORE NUMERICAL IMPORTS)
# ==============================================================================
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd


NPZ_PATH = "/mnt/data3/veasl001/TEMPO_Replication/data/processed/epa_point_dataset_14months_20features.npz"


def print_banner(text: str):
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def inspect_npz(file_path: str):
    if not os.path.exists(file_path):
        print(f"[!] Target NPZ dataset not found at: {file_path}")
        sys.exit(1)

    print_banner("1. NPZ FILE KEY RECONNAISSANCE")
    dataset = np.load(file_path, allow_pickle=True)
    keys = dataset.files
    print(f"Dataset Path : {file_path}")
    print(f"File Size    : {os.path.getsize(file_path) / (1024**2):.2f} MB")
    print(f"Keys Found   : {keys}")

    # Display shapes and dtypes
    print("\nArray Manifest:")
    for k in keys:
        arr = dataset[k]
        print(f"  - Key: '{k:<15}' | Shape: {str(arr.shape):<18} | Dtype: {arr.dtype}")

    # Decode Feature Names
    print_banner("2. FEATURE VECTOR MAPPING (20 COVARIATES)")
    if 'feature_names' in keys:
        feature_names = dataset['feature_names']
        print(f"Total Features Discovered: {len(feature_names)}")
        for i, name in enumerate(feature_names):
            print(f"  Index [{i:02d}] : {name}")
    else:
        print("[!] Key 'feature_names' missing from NPZ dictionary.")
        feature_names = [f"feature_{i}" for i in range(dataset['X_train'].shape[1])]

    # Group Array Metadata Evaluation
    print_banner("3. GROUPS ARRAY SEMANTIC ANALYSIS")
    groups_train = dataset['groups_train'] if 'groups_train' in keys else None
    groups_test = dataset['groups_test'] if 'groups_test' in keys else None

    if groups_train is not None and groups_test is not None:
        u_train = np.unique(groups_train)
        u_test = np.unique(groups_test)
        
        print(f"groups_train -> Length: {len(groups_train):,}, Unique Groups: {len(u_train)}")
        print(f"  Unique Values : {u_train}")
        print(f"  Value Counts  :")
        val, counts = np.unique(groups_train, return_counts=True)
        for v, c in zip(val, counts):
            print(f"    Group {v}: {c:,} samples ({c/len(groups_train):.2%})")

        print(f"\ngroups_test  -> Length: {len(groups_test):,}, Unique Groups: {len(u_test)}")
        print(f"  Unique Values : {u_test}")
        print(f"  Value Counts  :")
        val_t, counts_t = np.unique(groups_test, return_counts=True)
        for v, c in zip(val_t, counts_t):
            print(f"    Group {v}: {c:,} samples ({c/len(groups_test):.2%})")

        # Evaluate if groups correspond to K-Means Spatial Clusters (e.g. k=5)
        if len(u_train) <= 10:
            print("\n[EVALUATION]: 'groups' corresponds to Spatial K-Means Cluster IDs (0 through K-1).")
        else:
            print("\n[EVALUATION]: 'groups' corresponds to Station IDs or Temporal Block IDs.")

    # Feature Matrix Sample & Range Inspection
    print_banner("4. FEATURE MATRIX STATISTICAL PROFILE (X_train)")
    X_train = dataset['X_train']
    y_train = dataset['y_train']

    df_stats = pd.DataFrame({
        'Feature': feature_names,
        'Min': np.min(X_train, axis=0),
        'Mean': np.mean(X_train, axis=0),
        'Max': np.max(X_train, axis=0),
        'Std': np.std(X_train, axis=0),
        'NaN Count': np.isnan(X_train).sum(axis=0)
    })
    
    # Adjust pandas display limits
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(df_stats.to_string(index=True))

    print("\nTarget Variable (y_train - True NO2 Surface Mixing Ratio):")
    print(f"  Min: {np.min(y_train):.4f} ppb | Mean: {np.mean(y_train):.4f} ppb | Max: {np.max(y_train):.4f} ppb | Std: {np.std(y_train):.4f}")

    # Inspect for Station Coordinates / IDs
    print_banner("5. SPATIAL & TEMPORAL IDENTIFIER DETECTIVE")
    coord_cols = [name for name in feature_names if any(term in name.lower() for term in ['lat', 'lon', 'x', 'y', 'station', 'site', 'time', 'date', 'hour'])]
    
    if coord_cols:
        print(f"Discovered potential spatial/temporal columns in X matrix: {coord_cols}")
    else:
        print("[!] Explicit geographic coordinates (lat/lon) or raw station IDs were NOT found as named columns in X_train.")
        print("    -> Checking if positional encodings or sin/cos periodicities exist in features...")

    # Print First 3 Raw Feature Vectors
    print_banner("6. SAMPLE FEATURE VECTORS (First 3 Rows of X_train)")
    for r in range(min(3, X_train.shape[0])):
        print(f"\n--- Row {r} (y = {y_train[r]:.2f} ppb, Group = {groups_train[r]}) ---")
        for i, name in enumerate(feature_names):
            print(f"  {name:<22} : {X_train[r, i]:.6f}")

    print_banner("INSPECTION COMPLETE")


if __name__ == "__main__":
    inspect_npz(NPZ_PATH)