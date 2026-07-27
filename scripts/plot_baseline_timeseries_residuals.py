import os
from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "epa_from_internet_daily"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

print("📈 STARTING BASELINE TIME SERIES & RESIDUALS GENERATION 📈")
print("=" * 70)

# ==============================================================================
# 2. LOAD & COMBINE 2023-2024 MONITOR DATA
# ==============================================================================
files_to_load = [
    RAW_DIR / "daily_42602_2023.csv",
    RAW_DIR / "daily_42602_2024.csv",
]

dfs = []
for f_path in files_to_load:
  if not f_path.exists():
    f_path = BASE_DIR / "raw" / "epa_from_internet_daily" / f_path.name
  if f_path.exists():
    print(f"⏳ Loading {f_path.name}...")
    df = pd.read_csv(f_path, low_memory=False)
    dfs.append(df)
  else:
    raise FileNotFoundError(f"Could not find required data file: {f_path}")

df_all = pd.concat(dfs, ignore_index=True)
df_all["Date Local"] = pd.to_datetime(df_all["Date Local"])
df_all = df_all.sort_values("Date Local")

if df_all["Arithmetic Mean"].max() < 5.0:
  df_all["NO2_ppb"] = df_all["Arithmetic Mean"] * 1000.0
else:
  df_all["NO2_ppb"] = df_all["Arithmetic Mean"]

# ==============================================================================
# 3. FILTER FOR SELECTED SITES
# ==============================================================================
target_sites = ["Compton", "Pomona", "Santa Clarita"]
df_all["Site_Name_Clean"] = df_all["Local Site Name"].astype(str).str.strip()
df_sites = df_all[df_all["Site_Name_Clean"].isin(target_sites)].copy()

print(f"✅ Extracted data for: {target_sites}")
print("=" * 70)


# ==============================================================================
# 4. PLOTTING ENGINE FOR TIME SERIES & RESIDUALS
# ==============================================================================
def generate_timeseries_and_residuals(
    site_name, site_df, color="#4185c4", model_type="Mean"
):
  daily_ts = site_df.groupby("Date Local")["NO2_ppb"].mean().reset_index()
  dates = daily_ts["Date Local"]
  actual = daily_ts["NO2_ppb"].values
  n_days = len(actual)

  # Determine which baseline guess to use
  val_mean = np.mean(actual)
  val_median = np.median(actual)

  binned_data = np.round(actual * 2) / 2
  mode_res = stats.mode(binned_data, keepdims=True)
  val_mode = mode_res.mode[0] if len(mode_res.mode) > 0 else val_median

  if model_type == "Mean":
    pred_val = val_mean
  elif model_type == "Median":
    pred_val = val_median
  else:
    pred_val = val_mode

  predicted = np.full_like(actual, fill_value=pred_val)
  std_dev = np.std(actual)

  # Calculate Performance Metrics
  rmse = np.sqrt(mean_squared_error(actual, predicted))
  mae = mean_absolute_error(actual, predicted)
  r2 = r2_score(actual, predicted)

  # Calculate Residuals (Predicted - Actual, matching Slide 19 formula)
  residuals = predicted - actual
  res_mean = np.mean(residuals)
  res_std = np.std(residuals)
  res_min = np.min(residuals)
  res_max = np.max(residuals)

  # --------------------------------------------------------------------------
  # FIGURE 1: OBSERVED vs BASELINE TIME SERIES (MIRRORS SLIDE 15)
  # --------------------------------------------------------------------------
  fig1, ax1 = plt.subplots(figsize=(15, 8))

  # Plot Shaded ±1 Standard Deviation Band
  ax1.fill_between(
      dates,
      pred_val - std_dev,
      pred_val + std_dev,
      color="#f8d7da",
      alpha=0.6,
      label=f"±1 std dev ({std_dev:.2f} ppb)",
      zorder=1,
  )

  # Plot Observed Time Series
  ax1.plot(
      dates,
      actual,
      color=color,
      linewidth=1.2,
      alpha=0.85,
      label="Observed NO₂",
      zorder=2,
  )

  # Plot Constant Baseline Model Line
  ax1.axhline(
      pred_val,
      color="#d62728",
      linestyle="--",
      linewidth=2.2,
      label=f"{model_type} Model ({pred_val:.2f} ppb)",
      zorder=3,
  )

  # Formatting Fig 1
  ax1.set_ylabel("NO₂ Daily Mean (ppb)", fontsize=12, fontweight="bold")
  ax1.set_xlabel("Date", fontsize=12, fontweight="bold")
  ax1.set_ylim(0, max(actual) * 1.15)
  ax1.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=0)
  ax1.set_title(
      f"{model_type} Model vs Observed NO₂ - {site_name} Site (2023–2024)",
      fontsize=15,
      fontweight="bold",
      pad=15,
  )
  ax1.legend(
      loc="upper right", frameon=True, facecolor="white", edgecolor="gray", fontsize=10.5
  )

  # Top-Left Performance Overlay Box
  perf_box1 = (
      f"{model_type} Model\n"
      f"Prediction: {pred_val:.2f} ppb (constant)\n"
      f"Std Dev:    {std_dev:.2f} ppb\n\n"
      "Overall Performance:\n"
      f"  RMSE: {rmse:.2f} ppb\n"
      f"  MAE:  {mae:.2f} ppb\n"
      f"  R²:   {r2:.3f}\n\n"
      f"N = {n_days} days"
  )
  ax1.text(
      0.015,
      0.96,
      perf_box1,
      transform=ax1.transAxes,
      fontsize=9.5,
      verticalalignment="top",
      fontfamily="monospace",
      bbox=dict(
          boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="gray", alpha=0.95
      ),
      zorder=4,
  )

  # Bottom-Right Explanatory Box
  note_box1 = (
      f"Baseline Model:\nPredicts overall {model_type.lower()}\nfor every day\nEPA"
      " AQS Data 2023-2024"
  )
  ax1.text(
      0.985,
      0.04,
      note_box1,
      transform=ax1.transAxes,
      fontsize=8.5,
      verticalalignment="bottom",
      horizontalalignment="right",
      bbox=dict(
          boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9
      ),
      zorder=4,
  )

  # Date Formatting
  ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
  ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
  plt.xticks(fontsize=10, fontweight="bold")

  clean_site = site_name.lower().replace(" ", "_")
  out_file1 = (
      OUTPUT_DIR / f"timeseries_observed_vs_{model_type.lower()}_{clean_site}.png"
  )
  plt.savefig(out_file1, dpi=300, bbox_inches="tight")
  plt.close(fig1)

  # --------------------------------------------------------------------------
  # FIGURE 2: TIME SERIES OF RESIDUALS (MIRRORS SLIDE 19)
  # --------------------------------------------------------------------------
  fig2, ax2 = plt.subplots(figsize=(15, 8))

  # Plot Shaded ±1 Residual Standard Deviation Band
  ax2.fill_between(
      dates,
      -res_std,
      res_std,
      color="#f8d7da",
      alpha=0.6,
      label=f"±1 Std Dev (±{res_std:.2f} ppb)",
      zorder=1,
  )

  # Plot Solid Zero Line
  ax2.axhline(
      0,
      color="#555555",
      linestyle="-",
      linewidth=1.8,
      label="Zero Line (Perfect Guess)",
      zorder=2,
  )

  # Plot Residuals Time Series
  ax2.plot(
      dates,
      residuals,
      color="#2b5c8f",
      linewidth=1.1,
      alpha=0.9,
      label="Residuals (Predicted - Actual)",
      zorder=3,
  )

  # Formatting Fig 2
  ax2.set_ylabel(
      "Residual (Predicted - Actual) [ppb]", fontsize=12, fontweight="bold"
  )
  ax2.set_xlabel("Date", fontsize=12, fontweight="bold")
  ax2.set_ylim(min(residuals) * 1.15, max(residuals) * 1.35)
  ax2.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=0)
  ax2.set_title(
      f"{model_type} Model Residuals - {site_name} Site (2023–2024)\n(Predicted -"
      " Actual NO₂)",
      fontsize=15,
      fontweight="bold",
      pad=15,
  )
  ax2.legend(
      loc="upper right", frameon=True, facecolor="white", edgecolor="gray", fontsize=10.5
  )

  # Top-Left Performance Overlay Box
  perf_box2 = (
      f"{model_type} Model Residuals\n\n"
      f"Model Guess:   {pred_val:.2f} ppb\n"
      f"Residual mean: {res_mean:.4f} ppb\n"
      f"Residual std:  {res_std:.2f} ppb\n\n"
      "Performance:\n"
      f"  RMSE: {rmse:.2f} ppb\n"
      f"  MAE:  {mae:.2f} ppb\n\n"
      "Residual range:\n"
      f"  Min: {res_min:6.2f} ppb\n"
      f"  Max: {res_max:6.2f} ppb"
  )
  ax2.text(
      0.015,
      0.96,
      perf_box2,
      transform=ax2.transAxes,
      fontsize=9.5,
      verticalalignment="top",
      fontfamily="monospace",
      bbox=dict(
          boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="gray", alpha=0.95
      ),
      zorder=4,
  )

  # Bottom-Right Note Box
  note_box2 = (
      "Note:\n"
      "Positive residuals = overprediction\n"
      "Negative residuals = underprediction\n"
      "Good models have residuals centered\n"
      "around zero with minimal systematic patterns"
  )
  ax2.text(
      0.985,
      0.04,
      note_box2,
      transform=ax2.transAxes,
      fontsize=8.5,
      verticalalignment="bottom",
      horizontalalignment="right",
      bbox=dict(
          boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9
      ),
      zorder=4,
  )

  # Date Formatting
  ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
  ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
  plt.xticks(fontsize=10, fontweight="bold")

  out_file2 = OUTPUT_DIR / f"residuals_{model_type.lower()}_model_{clean_site}.png"
  plt.savefig(out_file2, dpi=300, bbox_inches="tight")
  plt.close(fig2)

  print(f"   ✅ {model_type:6s} Model | Saved Time Series & Residual plots!")


# ==============================================================================
# 5. EXECUTE ACROSS ALL SITES AND ALL THREE BASELINE MODELS
# ==============================================================================
colors = {
    "Compton": "#d62728",  # Deep Red
    "Pomona": "#ff7f0e",  # Orange
    "Santa Clarita": "#1f77b4",  # Royal Blue
}

for site in target_sites:
  print(f"\n📍 Generating Plots for Site: {site}...")
  site_data = df_sites[df_sites["Site_Name_Clean"] == site].copy()

  # Generate for Mean, Median, and Mode baselines!
  for m_type in ["Mean", "Median", "Mode"]:
    generate_timeseries_and_residuals(
        site, site_data, color=colors[site], model_type=m_type
    )

print("\n🎯 ALL TIME SERIES & RESIDUAL GENERATIONS SUCCESSFULLY FINISHED!")