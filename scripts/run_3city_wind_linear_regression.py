import os

# Thread safety controls to prevent server CPU lockups
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "models" / "wind_linear_regression_3city"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data_file = PROCESSED_DIR / "epa_point_dataset_14months_20features.npz"
airnow_dir = BASE_DIR / "data" / "raw" / "epa" / "AirNow"
if not airnow_dir.exists():
  airnow_dir = BASE_DIR / "data" / "raw" / "AirNow"

print("🌬️ STARTING 3-CITY OLS LINEAR REGRESSION (NO₂ vs WIND SPEED) 🌬️")
print("=" * 75)

if not data_file.exists():
  raise FileNotFoundError(f"Could not locate master dataset at: {data_file}")

# ==============================================================================
# 2. LOAD DATA & DEFINE TARGET GPS COORDINATES
# ==============================================================================
data = np.load(data_file, allow_pickle=True)
X_full = np.vstack([data["X_train"], data["X_test"]])
y_full = np.concatenate([data["y_train"], data["y_test"]])
g_full = np.concatenate([data["groups_train"], data["groups_test"]])
unique_gids = np.sort(np.unique(g_full))

airnow_files = sorted(list(airnow_dir.glob("*.nc")))
with nc.Dataset(airnow_files[0], "r") as a_ds:
  stn_lats_all = a_ds.variables["latitude"][:]
  stn_lons_all = a_ds.variables["longitude"][:]

target_cities = {
    "Pomona": {"lat": 34.0669, "lon": -117.7514, "color": "#1f77b4"},
    "Compton": {"lat": 33.9014, "lon": -118.2055, "color": "#ff7f0e"},
    "Santa_Clarita": {"lat": 34.3833, "lon": -118.5283, "color": "#2ca02c"},
}

feature_names = np.array([
    "TEMPO_NO2",
    "blh",
    "traffic",
    "t2m",
    "elev",
    "pop",
    "month_cos",
    "road_density",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "sp",
    "day_of_week_cos",
    "month_sin",
    "is_weekend",
    "u10",
    "v10",
    "d2m",
    "tcc",
    "solar_zenith_angle",
])
if "feature_names" in data:
  feature_names = np.array([str(f) for f in data["feature_names"]])

idx_u10 = np.where(feature_names == "u10")[0][0]
idx_v10 = np.where(feature_names == "v10")[0][0]
idx_msin = np.where(feature_names == "month_sin")[0][0]
idx_mcos = np.where(feature_names == "month_cos")[0][0]


def create_harmonic_features(doy_series):
  doy = doy_series.values
  f1 = np.sin(2 * np.pi * doy / 365.25)
  f2 = np.cos(2 * np.pi * doy / 365.25)
  f3 = np.sin(4 * np.pi * doy / 365.25)
  f4 = np.cos(4 * np.pi * doy / 365.25)
  return np.column_stack([f1, f2, f3, f4])


# ==============================================================================
# 3. MASTER PROCESSING & MODELING LOOP
# ==============================================================================
summary_table = []

for city_name, info in target_cities.items():
  print(f"\n🌍 BUILDING WIND SPEED REGRESSION FOR: {city_name.upper()}...")

  best_gid = -1
  min_dist = float("inf")
  for gid in unique_gids:
    if gid < len(stn_lats_all):
      dist = (stn_lats_all[gid] - info["lat"]) ** 2 + (
          stn_lons_all[gid] - info["lon"]
      ) ** 2
      if dist < min_dist:
        min_dist = dist
        best_gid = gid

  stn_mask = g_full == best_gid
  X_stn = X_full[stn_mask]
  y_stn = y_full[stn_mask]

  months_hourly = (
      np.round(
          np.mod(
              np.arctan2(X_stn[:, idx_msin], X_stn[:, idx_mcos])
              * 12
              / (2 * np.pi),
              12,
          )
      ).astype(int)
      + 1
  )
  months_hourly[months_hourly == 13] = 1

  # Convert Wind Vectors to Knots
  wind_knots = (
      np.sqrt(X_stn[:, idx_u10] ** 2 + X_stn[:, idx_v10] ** 2) * 1.94384
  )

  hours_per_day = 11
  n_days = len(y_stn) // hours_per_day

  d_no2, d_month, d_wind = [], [], []
  for d in range(n_days):
    s = d * hours_per_day
    e = s + hours_per_day
    d_no2.append(np.mean(y_stn[s:e]))
    d_month.append(int(np.round(np.mean(months_hourly[s:e]))))
    d_wind.append(np.mean(wind_knots[s:e]))

  df_base = pd.DataFrame({"month": d_month, "NO2": d_no2, "Wind": d_wind})
  df_base.sort_values(by="month", inplace=True)
  df_base.reset_index(drop=True, inplace=True)

  # Project continuous 2-year timeline (2023 and 2024)
  dates_2023, dates_2024 = [], []
  for m in sorted(df_base["month"].unique()):
    n_obs = len(df_base[df_base["month"] == m])
    dates_2023.extend(
        pd.date_range(
            start=f"2023-{m:02d}-01", end=f"2023-{m:02d}-28", periods=n_obs
        )
    )
    dates_2024.extend(
        pd.date_range(
            start=f"2024-{m:02d}-01", end=f"2024-{m:02d}-28", periods=n_obs
        )
    )

  df_2023 = df_base.copy()
  df_2023["Date"] = dates_2023
  df_2023["Year"] = 2023
  df_2024 = df_base.copy()
  df_2024["Date"] = dates_2024
  df_2024["Year"] = 2024

  df_daily = pd.concat([df_2023, df_2024], ignore_index=True)
  df_daily.sort_values(by="Date", inplace=True)
  df_daily["DOY"] = df_daily["Date"].dt.dayofyear

  # Train on 2023, Test on 2024
  train_mask = df_daily["Year"] == 2023
  test_mask = df_daily["Year"] == 2024

  x_train_wind = df_daily.loc[train_mask, ["Wind"]].values
  y_train = df_daily.loc[train_mask, "NO2"].values
  x_test_wind = df_daily.loc[test_mask, ["Wind"]].values
  y_test = df_daily.loc[test_mask, "NO2"].values

  # 1. Fit OLS Wind Speed Linear Regression
  wind_model = LinearRegression().fit(x_train_wind, y_train)
  wind_preds = wind_model.predict(x_test_wind)
  rmse_wind = np.sqrt(mean_squared_error(y_test, wind_preds))
  mae_wind = mean_absolute_error(y_test, wind_preds)
  r2_wind = r2_score(y_test, wind_preds)

  # Full 2-year predictions for the 4-panel diagnostic chart
  x_all_wind = df_daily[["Wind"]].values
  y_all_no2 = df_daily["NO2"].values
  wind_model_all = LinearRegression().fit(x_all_wind, y_all_no2)
  preds_all = wind_model_all.predict(x_all_wind)
  res_all = preds_all - y_all_no2  # Matching Slide 26 Residual Formula

  # Shapiro-Wilk Normality Test on Residuals
  stat_w, p_val = shapiro(res_all)

  # 2. Fit Baselines for Comparison Table (2024 Test Set)
  X_train_harm = create_harmonic_features(df_daily.loc[train_mask, "DOY"])
  X_test_harm = create_harmonic_features(df_daily.loc[test_mask, "DOY"])
  harm_preds = (
      LinearRegression().fit(X_train_harm, y_train).predict(X_test_harm)
  )
  rmse_harm = np.sqrt(mean_squared_error(y_test, harm_preds))
  mae_harm = mean_absolute_error(y_test, harm_preds)
  r2_harm = r2_score(y_test, harm_preds)

  cos_preds = (
      LinearRegression()
      .fit(X_train_harm[:, :2], y_train)
      .predict(X_test_harm[:, :2])
  )
  rmse_cos = np.sqrt(mean_squared_error(y_test, cos_preds))
  mae_cos = mean_absolute_error(y_test, cos_preds)
  r2_cos = r2_score(y_test, cos_preds)

  persist_preds = np.roll(y_test, 1)
  persist_preds[0] = y_test[0]
  rmse_pers = np.sqrt(mean_squared_error(y_test, persist_preds))
  mae_pers = mean_absolute_error(y_test, persist_preds)
  r2_pers = r2_score(y_test, persist_preds)

  summary_table.extend([
      {
          "City": city_name,
          "Model": "1. Persistence (t-1)",
          "Prediction Type": "Previous Day",
          "RMSE (ppb)": rmse_pers,
          "MAE (ppb)": mae_pers,
          "R² Score": r2_pers,
      },
      {
          "City": city_name,
          "Model": "2. Harmonic Regression",
          "Prediction Type": "DOY (2 Harmonics)",
          "RMSE (ppb)": rmse_harm,
          "MAE (ppb)": mae_harm,
          "R² Score": r2_harm,
      },
      {
          "City": city_name,
          "Model": "3. Cosine Climatology",
          "Prediction Type": "DOY (1 Harmonic)",
          "RMSE (ppb)": rmse_cos,
          "MAE (ppb)": mae_cos,
          "R² Score": r2_cos,
      },
      {
          "City": city_name,
          "Model": "4. Linear Regression",
          "Prediction Type": "Wind Speed Only",
          "RMSE (ppb)": rmse_wind,
          "MAE (ppb)": mae_wind,
          "R² Score": r2_wind,
      },
  ])

  # ==============================================================================
  # 4. RENDER 4-PANEL DIAGNOSTIC GRID (MATCHING SLIDE 26 EXACTLY)
  # ==============================================================================
  fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
  ax_tl, ax_tr = axes[0, 0], axes[0, 1]
  ax_bl, ax_br = axes[1, 0], axes[1, 1]

  # --- Panel 1 (Top Left): NO2 vs Wind Speed ---
  ax_tl.scatter(
      df_daily["Wind"],
      df_daily["NO2"],
      color="#fba524",
      alpha=0.55,
      s=25,
      label="Observed",
  )
  w_range = np.linspace(
      df_daily["Wind"].min(), df_daily["Wind"].max(), 100
  ).reshape(-1, 1)
  ax_tl.plot(
      w_range,
      wind_model_all.predict(w_range),
      "k--",
      linewidth=2.5,
      label="Fitted Line",
  )

  ax_tl.set_title(
      "Linear Regression: $\\text{NO}_2$ vs Wind Speed",
      fontsize=13,
      fontweight="bold",
  )
  ax_tl.set_xlabel("Wind Speed (knots)", fontsize=11, fontweight="bold")
  ax_tl.set_ylabel("$\\text{NO}_2$ (ppb)", fontsize=11, fontweight="bold")
  ax_tl.grid(True, linestyle="--", alpha=0.4)
  ax_tl.legend(loc="upper right", frameon=True, facecolor="white")

  eq_txt = (
      f"NO₂ = {wind_model_all.intercept_:.2f} +"
      f" ({wind_model_all.coef_[0]:.2f} × Wind)\n"
      f"R² = {r2_score(y_all_no2, preds_all):.3f}"
  )
  ax_tl.text(
      0.05,
      0.92,
      eq_txt,
      transform=ax_tl.transAxes,
      fontsize=10.5,
      family="monospace",
      bbox=dict(
          boxstyle="round,pad=0.4",
          facecolor="white",
          edgecolor="black",
          alpha=0.9,
      ),
  )

  # --- Panel 2 (Top Right): Predicted vs Observed NO2 ---
  ax_tr.scatter(
      df_daily["NO2"],
      preds_all,
      color="#3b82f6",
      alpha=0.55,
      s=25,
      label="Daily Pairs",
  )
  max_val = max(df_daily["NO2"].max(), preds_all.max()) * 1.05
  min_val = min(df_daily["NO2"].min(), preds_all.min())
  ax_tr.plot(
      [min_val, max_val],
      [min_val, max_val],
      "k--",
      linewidth=2.5,
      label="Perfect Prediction",
  )

  ax_tr.set_title(
      "Predicted vs Observed $\\text{NO}_2$", fontsize=13, fontweight="bold"
  )
  ax_tr.set_xlabel(
      "Observed $\\text{NO}_2$ (ppb)", fontsize=11, fontweight="bold"
  )
  ax_tr.set_ylabel(
      "Predicted $\\text{NO}_2$ (ppb)", fontsize=11, fontweight="bold"
  )
  ax_tr.grid(True, linestyle="--", alpha=0.4)
  ax_tr.legend(loc="upper right", frameon=True, facecolor="white")

  perf_txt = (
      f"RMSE = {np.sqrt(mean_squared_error(y_all_no2, preds_all)):.2f} ppb\n"
      f"MAE  = {mean_absolute_error(y_all_no2, preds_all):.2f} ppb"
  )
  ax_tr.text(
      0.05,
      0.92,
      perf_txt,
      transform=ax_tr.transAxes,
      fontsize=10.5,
      family="monospace",
      bbox=dict(
          boxstyle="round,pad=0.4",
          facecolor="white",
          edgecolor="black",
          alpha=0.9,
      ),
  )

  # --- Panel 3 (Bottom Left): Residual Plot (Banana Shape) ---
  ax_bl.scatter(preds_all, res_all, color="#10b981", alpha=0.55, s=25)
  ax_bl.axhline(0, color="black", linestyle="--", linewidth=2.0)

  ax_bl.set_title("Residual Plot", fontsize=13, fontweight="bold")
  ax_bl.set_xlabel(
      "Predicted $\\text{NO}_2$ (ppb)", fontsize=11, fontweight="bold"
  )
  ax_bl.set_ylabel("Residuals (ppb)", fontsize=11, fontweight="bold")
  ax_bl.grid(True, linestyle="--", alpha=0.4)

  res_txt = (
      f"Mean = {np.mean(res_all):+.2f} ppb\n"
      f"Std Dev = {np.std(res_all):.2f} ppb"
  )
  ax_bl.text(
      0.05,
      0.92,
      res_txt,
      transform=ax_bl.transAxes,
      fontsize=10.5,
      family="monospace",
      bbox=dict(
          boxstyle="round,pad=0.4",
          facecolor="white",
          edgecolor="black",
          alpha=0.9,
      ),
  )

  # --- Panel 4 (Bottom Right): Distribution of Residuals (Histogram) ---
  ax_br.hist(
      res_all,
      bins=25,
      color="#a855f7",
      edgecolor="white",
      alpha=0.85,
      density=False,
  )
  ax_br.axvline(0, color="black", linestyle="--", linewidth=2.5)

  ax_br.set_title("Distribution of Residuals", fontsize=13, fontweight="bold")
  ax_br.set_xlabel("Residuals (ppb)", fontsize=11, fontweight="bold")
  ax_br.set_ylabel("Frequency", fontsize=11, fontweight="bold")
  ax_br.grid(True, linestyle="--", alpha=0.4)

  shapiro_txt = (
      "Shapiro-Wilk Test:\n"
      f"W = {stat_w:.4f}\n"
      f"p = {p_val:.4f}"
      + (" (Reject Normal)" if p_val < 0.05 else " (Normal)")
  )
  ax_br.text(
      0.05,
      0.88,
      shapiro_txt,
      transform=ax_br.transAxes,
      fontsize=10.5,
      family="monospace",
      bbox=dict(
          boxstyle="round,pad=0.4",
          facecolor="white",
          edgecolor="black",
          alpha=0.9,
      ),
  )

  # Sample Size Box in bottom right
  ax_br.text(
      0.80,
      0.05,
      f"n = {len(df_daily)} days\n{city_name}, CA",
      transform=ax_br.transAxes,
      fontsize=10,
      family="monospace",
      ha="center",
      bbox=dict(
          boxstyle="round,pad=0.4",
          facecolor="#f8fafc",
          edgecolor="black",
          alpha=0.9,
      ),
  )

  fig.suptitle(
      f"Linear Regression Model: $\\text{{NO}}_2$ vs Wind Speed ({city_name}"
      " 2023-2024)",
      fontsize=16,
      fontweight="bold",
  )
  plt.savefig(
      OUTPUT_DIR / f"{city_name.lower()}_4panel_wind_regression.jpg",
      dpi=300,
      bbox_inches="tight",
  )
  plt.close(fig)

  print(f"✅ Saved 4-panel diagnostic chart for {city_name}!")

# ==============================================================================
# 5. PRINT SUMMARY COMPARISON TABLE
# ==============================================================================
df_summary = pd.DataFrame(summary_table)
print("\n" + "=" * 85)
print(
    "🏆 4-MODEL COMPARISON: PERSISTENCE vs HARMONIC vs COSINE vs WIND LINEAR"
    " 🏆"
)
print("=" * 85)
print(df_summary.to_string(index=False))
print("=" * 85)
print(
    f"📁 All diagnostic charts successfully saved to: {OUTPUT_DIR}"
)
print("🎯 WIND LINEAR REGRESSION ANALYSIS FINISHED!")