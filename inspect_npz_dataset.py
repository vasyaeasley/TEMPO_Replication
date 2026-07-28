#!/usr/bin/env python3
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

NPZ_PATH = "/mnt/data3/veasl001/TEMPO_Replication/data/processed/epa_point_dataset_14months_20features.npz"

def inspect_npz(file_path: str):
    if not os.path.exists(file_path):
        print(f"[!] Target NPZ dataset not found at: {file_path}")
        sys.exit(1)

    print("=" * 80)
    print(" 1. NPZ FILE KEY RECONNAISSANCE")
    print("=" * 80)
    dataset = np.load(file_path, allow_pickle=True)
    keys = dataset.files
    print(f"Dataset Path : {file_path}")
    print(f"Keys Found   : {keys}")

    for k in keys:
        arr = dataset[k]
        print(f"  - Key: '{k:<15}' | Shape: {str(arr.shape):<18} | Dtype: {arr.dtype}")

    print("\n" + "=" * 80)
    print(" 2. FEATURE VECTOR MAPPING (20 COVARIATES)")
    print("=" * 80)
    feature_names = dataset['feature_names'] if 'feature_names' in keys else [f"feature_{i}" for i in range(dataset['X_train'].shape[1])]
    for i, name in enumerate(feature_names):
        print(f"  Index [{i:02d}] : {name}")

    print("\n" + "=" * 80)
    print(" 3. GROUPS ARRAY SEMANTIC ANALYSIS")
    print("=" * 80)
    groups_train = dataset['groups_train'] if 'groups_train' in keys else None
    groups_test = dataset['groups_test'] if 'groups_test' in keys else None

    if groups_train is not None:
        u_train, c_train = np.unique(groups_train, return_counts=True)
        print(f"groups_train -> Length: {len(groups_train):,}, Unique Groups: {len(u_train)}")
        for v, c in zip(u_train, c_train):
            print(f"    Group {v}: {c:,} samples ({c/len(groups_train):.2%})")

    if groups_test is not None:
        u_test, c_test = np.unique(groups_test, return_counts=True)
        print(f"\ngroups_test  -> Length: {len(groups_test):,}, Unique Groups: {len(u_test)}")
        for v, c in zip(u_test, c_test):
            print(f"    Group {v}: {c:,} samples ({c/len(groups_test):.2%})")

    print("\n" + "=" * 80)
    print(" 4. FEATURE MATRIX STATISTICAL PROFILE (X_train)")
    print("=" * 80)
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
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: '%.4f' % x)
    print(df_stats.to_string(index=True))

    print("\n" + "=" * 80)
    print(" 5. FIRST 3 SAMPLE ROWS (X_train)")
    print("=" * 80)
    for r in range(min(3, X_train.shape[0])):
        print(f"\n--- Row {r} (y = {y_train[r]:.2f} ppb, Group = {groups_train[r] if groups_train is not None else 'N/A'}) ---")
        for i, name in enumerate(feature_names):
            print(f"  {name:<22} : {X_train[r, i]:.6f}")

if __name__ == "__main__":
    inspect_npz(NPZ_PATH)
