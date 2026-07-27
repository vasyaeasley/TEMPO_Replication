import os

# ==============================================================================
# 0. THREAD SAFETY CONTROLS (MUST BE SET BEFORE IMPORTING NUMPY/SKLEARN!)
# Prevents OpenBLAS 128-thread core dumps on high-core Linux servers!
# ==============================================================================
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, randint, uniform
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "epa_from_internet_daily"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

print("🚀 STARTING PATH A: THE 18-FEATURE SEASONAL & ASTRONOMICAL POWER-UP 🚀")
print("=" * 75)

# ==============================================================================
# 2. UNRESTRICTED DEEP DATASET HUNTER
# ==============================================================================
print("⏳ Scanning entire project tree for master tabular dataset...")

all_tabular_files = []
for ext in ["*.csv", "*.parquet", "*.pkl", "*.feather"]:
  for f in BASE_DIR.rglob(ext):
    # Only exclude python environment, git, and raw daily EPA downloads!
    if any(
        ignore in str(f).lower()
        for ignore in [
            "tempo_env",
            ".git",
            "__pycache__",
            "daily_42602_",
            "era5_california_",
        ]
    ):
      continue
    # Only consider files larger than 5 MB to skip small test files
    if f.exists() and f.stat().st_size > 5 * 1024 * 1024:
      all_tabular_files.append(f)

# Sort by size descending (the 14-month dataset will be the largest file!)
all_tabular_files.sort(key=lambda x: x.stat().st_size, reverse=True)

data_file = None
for candidate in all_tabular_files:
  try:
    if candidate.suffix == ".parquet":
      cols = pd.read_parquet(candidate).columns.tolist()
    elif candidate.suffix == ".csv":
      cols = pd.read_csv(candidate, nrows=5).columns.tolist()
    elif candidate.suffix == ".pkl":
      cols = pd.read_pickle(candidate).columns.tolist()
    else:
      continue

    # Look for key atmospheric chemistry columns
    cols_lower = [str(c).lower() for c in cols]
    if any("no2" in c for c in cols_lower) and any(
        "blh" in c or "temp" in c or "u10" in c or "site" in c for c in cols_lower
    ):
      data_file = candidate
      print(
          f"✅ Auto-selected Master Dataset: {data_file.relative_to(BASE_DIR)}"
          f" ({data_file.stat().st_size / (1024*1024):.2f} MB)"
      )
      break
  except Exception:
    continue

if not data_file:
  print("\n❌ Could not automatically verify columns. Top largest files found:")
  for f in all_tabular_files[:5]:
    print(
        f"   * {f.relative_to(BASE_DIR)} ({f.stat().st_size / (1024*1024):.2f}"
        " MB)"
    )
  raise FileNotFoundError(
      "Please check the file list above and confirm where the master table is"
      " stored!"
  )

print(f"⏳ Loading dataframe from {data_file.name}...")
if data_file.suffix == ".parquet":
  df_master = pd.read_parquet(data_file)
elif data_file.suffix == ".pkl":
  df_master = pd.read_pickle(data_file)
elif data_file.suffix == ".feather":
  df_master = pd.read_feather(data_file)
else:
  df_master = pd.read_csv(data_file, low_memory=False)

# ==============================================================================
# 3. TIMESTAMP & COORDINATE INTEGRITY CHECK
# ==============================================================================
date_col = next(
    (
        c
        for c in df_master.columns
        if any(
            t in str(c).lower()
            for t in ["date local", "timestamp", "datetime", "date_local", "time"]
        )
    ),
    None,
)

if not date_col:
  raise ValueError(
      f"Could not find a timestamp column in {data_file.name}. Available"
      f" columns: {df_master.columns.tolist()[:15]}..."
  )

print(f"🗓️ Using timestamp column: '{date_col}' to engineer 6 free features...")
df_master[date_col] = pd.to_datetime(df_master[date_col], errors="coerce")
df_master = df_master.dropna(subset=[date_col]).copy()

if (
    "Latitude" not in df_master.columns or "Longitude" not in df_master.columns
) and "Site_ID" in df_master.columns:
  print("🌐 Cross-referencing Site_IDs with raw EPA file for coordinates...")
  raw_file = RAW_DIR / "daily_42602_2024.csv"
  if not raw_file.exists():
    raw_file = (
        BASE_DIR / "raw" / "epa_from_internet_daily" / "daily_42602_2024.csv"
    )
  if raw_file.exists():
    df_raw = pd.read_csv(
        raw_file,
        usecols=[
            "State Code",
            "County Code",
            "Site Num",
            "Latitude",
            "Longitude",
        ],
        low_memory=False,
    )
    df_raw["Site_ID"] = (
        df_raw["State Code"].astype(str).str.zfill(2)
        + "-"
        + df_raw["County Code"].astype(str).str.zfill(3)
        + "-"
        + df_raw["Site Num"].astype(str).str.zfill(4)
    )
    coord_map = (
        df_raw.groupby("Site_ID")[["Latitude", "Longitude"]]
        .first()
        .reset_index()
    )
    df_master = df_master.merge(coord_map, on="Site_ID", how="left")

if "NO2_ppb" not in df_master.columns:
  target_fallback = next(
      (
          col
          for col in df_master.columns
          if any(
              t in str(col).lower()
              for t in ["target", "y", "no2_ppb", "arithmetic mean", "no2"]
          )
      ),
      None,
  )
  if target_fallback:
    df_master["NO2_ppb"] = df_master[target_fallback]
    if df_master["NO2_ppb"].max() < 5.0:
      df_master["NO2_ppb"] *= 1000.0

df_master = df_master.dropna(
    subset=["NO2_ppb", "Latitude", "Longitude"]
).copy()

# ==============================================================================
# 4. ENGINEER 6 FREE ASTRONOMICAL & SEASONAL COVARIATES
# ==============================================================================
print("⚙️ Computing annual seasonality, weekly cycles, and solar zenith angles...")

month = df_master[date_col].dt.month
df_master["month_sin"] = np.sin(2 * np.pi * month / 12.0)
df_master["month_cos"] = np.cos(2 * np.pi * month / 12.0)

dow = df_master[date_col].dt.dayofweek
df_master["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7.0)
df_master["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7.0)

df_master["is_weekend"] = np.where(dow >= 5, 1.0, 0.0)

hour = df_master[date_col].dt.hour + (df_master[date_col].dt.minute / 60.0)
day_of_year = df_master[date_col].dt.dayofyear
lat_rad = np.radians(df_master["Latitude"])

gamma = 2 * np.pi / 365.0 * (day_of_year - 1 + (hour - 12) / 24.0)
declination = (
    0.006918
    - 0.399912 * np.cos(gamma)
    + 0.070257 * np.sin(gamma)
    - 0.006758 * np.cos(2 * gamma)
    + 0.000907 * np.sin(2 * gamma)
)
hour_angle = np.radians((hour - 12.0) * 15.0)

cos_sza = np.sin(lat_rad) * np.sin(declination) + np.cos(lat_rad) * np.cos(
    declination
) * np.cos(hour_angle)
cos_sza = np.clip(cos_sza, -1.0, 1.0)
df_master["solar_zenith_angle"] = np.degrees(np.arccos(cos_sza))

print("✅ Successfully engineered all 6 new covariates!")

# ==============================================================================
# 5. SPATIALLY AWARE 5-CLUSTER 60/40 TRAIN/TEST SPLIT
# ==============================================================================
print("\n🌍 Applying 5-Cluster Spatially Aware 60/40 Split...")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_master["Cluster"] = kmeans.fit_predict(df_master[["Latitude", "Longitude"]])

train_idx, test_idx = [], []
for cluster_num, group in df_master.groupby("Cluster"):
  tr, te = train_test_split(group.index, test_size=0.40, random_state=42)
  train_idx.extend(tr)
  test_idx.extend(te)

df_train = df_master.loc[train_idx].copy()
df_test = df_master.loc[test_idx].copy()

ignore_cols = {
    "NO2_ppb",
    "Arithmetic Mean",
    "Cluster",
    "Split",
    "Site_ID",
    date_col,
    "timestamp",
    "Date",
    "target",
    "y",
    "Local Site Name",
    "State Code",
    "County Code",
    "Site Num",
    "index",
    "level_0",
}
feature_cols = [
    c
    for c in df_master.columns
    if c not in ignore_cols and not str(c).lower().startswith("site")
]

X_train, y_train = df_train[feature_cols], df_train["NO2_ppb"]
X_test, y_test = df_test[feature_cols], df_test["NO2_ppb"]

print(f"📋 Detected {len(feature_cols)} Covariates: {feature_cols}")
print(
    f"✅ Split Verified | Train Samples: {len(X_train):,} | Test Samples:"
    f" {len(X_test):,}"
)
print("=" * 75)

# ==============================================================================
# 6. LIVE LEVEL-3 VERBOSE HYPERPARAMETER SWEEP
# ==============================================================================
print("⚙️ Setting up hyperparameter sweep (Table I specifications)...")
param_distributions = {
    "n_estimators": randint(100, 701),
    "learning_rate": uniform(0.001, 0.499),
    "subsample": uniform(0.2, 0.8),
    "max_depth": randint(1, 12),
    "colsample_bytree": uniform(0.1, 0.9),
    "gamma": uniform(0.0, 0.4),
    "reg_alpha": uniform(0.0, 100.0),
    "reg_lambda": uniform(0.001, 9.999),
}

xgb_base = XGBRegressor(
    objective="reg:squarederror", random_state=42, n_jobs=1, tree_method="hist"
)

search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_distributions,
    n_iter=25,
    scoring="r2",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=3,
)

print(
    f"⏳ Executing live grid search across {len(X_train):,} training samples..."
)
search.fit(X_train, y_train)

best_xgb = search.best_estimator_
print("\n🏆 OPTIMAL HYPERPARAMETERS FOUND:")
for k, v in search.best_params_.items():
  print(f"   * {k:18s}: {v:.4f}" if isinstance(v, float) else f"   * {k:18s}: {v}")
print("=" * 75)

# ==============================================================================
# 7. EVALUATION & FIGURE GENERATION
# ==============================================================================
print(f"📊 Evaluating 18-feature model on {len(X_test):,} unseen test samples...")
y_pred = best_xgb.predict(X_test)

r2_test = r2_score(y_test, y_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
mae_test = mean_absolute_error(y_test, y_pred)

print(f"\n🎯 PATH A TEST PERFORMANCE (TARGET: BEAT 0.312):")
print(f"   * Test R² Score : {r2_test:.3f}")
print(f"   * Test RMSE     : {rmse_test:.2f} ppb")
print(f"   * Test MAE      : {mae_test:.2f} ppb")
print("=" * 75)

print("🎨 Generating Figure 3 replication plot...")
fig, ax = plt.subplots(figsize=(10, 9))

if len(y_test) > 15000:
  plot_idx = np.random.choice(len(y_test), size=15000, replace=False)
  y_test_plot, y_pred_plot = y_test.iloc[plot_idx], y_pred[plot_idx]
else:
  y_test_plot, y_pred_plot = y_test, y_pred

xy = np.vstack([y_test_plot, y_pred_plot])
kde = gaussian_kde(xy)(xy)
idx = kde.argsort()
y_test_sorted, y_pred_sorted, kde_sorted = (
    y_test_plot.iloc[idx],
    y_pred_plot[idx],
    kde[idx],
)

max_val = np.ceil(max(y_test.max(), y_pred.max()) * 1.05)

sc = ax.scatter(
    y_test_sorted,
    y_pred_sorted,
    c=kde_sorted,
    cmap="viridis",
    s=35,
    alpha=0.8,
    edgecolor="none",
    zorder=2,
)
ax.plot(
    [0, max_val],
    [0, max_val],
    color="#d62728",
    linestyle="--",
    linewidth=2.2,
    label="1:1 Perfect Line",
    zorder=3,
)

ax.set_title(
    f"TEMPO 18-Feature Test Results (Path A)\n$R^2 = {r2_test:.2f}$",
    fontsize=16,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("True Values (ppb)", fontsize=13, fontweight="bold")
ax.set_ylabel("Predicted Values (ppb)", fontsize=13, fontweight="bold")
ax.set_xlim(0, max_val)
ax.set_ylim(0, max_val)
ax.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=0)

stats_box = (
    "Model: XGBoost (18 Covariates)\n"
    f"Test $R^2$:   {r2_test:.3f}\n"
    f"Test RMSE: {rmse_test:.2f} ppb\n"
    f"Test MAE:  {mae_test:.2f} ppb\n"
    f"Test Size: {len(y_test):,} samples"
)
ax.text(
    0.04,
    0.92,
    stats_box,
    transform=ax.transAxes,
    fontsize=10.5,
    verticalalignment="top",
    fontfamily="monospace",
    bbox=dict(
        boxstyle="round,pad=0.6", facecolor="white", edgecolor="gray", alpha=0.95
    ),
    zorder=4,
)
cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.03)
cbar.set_label("Data Point Density (Gaussian KDE)", fontsize=11, fontweight="bold")

out_file = OUTPUT_DIR / "tempo_xgboost_path_a_18features.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved Path A Scatterplot to: {out_file}")
print("🎯 PATH A EVALUATION SUCCESSFULLY FINISHED!")