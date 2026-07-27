import os
from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "epa_from_internet_daily"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

print("🦹‍♂️ STARTING PERSISTENT MODEL ('THE NEMESIS') EVALUATION 🦹‍♂️")
print("=" * 75)

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
print("=" * 75)

# ==============================================================================
# 4. GENERATE 1x3 SCATTERPLOT COMPARISON (MIRRORS SLIDE 22)
# ==============================================================================
print("🎨 Generating 1x3 Persistent Model Scatterplot Comparison...")
fig_scat, axes_scat = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig_scat.subplots_adjust(wspace=0.15)

colors = {
    "Compton": "#d62728",  # Deep Red
    "Pomona": "#ff7f0e",  # Orange
    "Santa Clarita": "#1f77b4",  # Royal Blue
}

metrics_summary = {}

for idx, site in enumerate(target_sites):
  ax = axes_scat[idx]
  site_df = df_sites[df_sites["Site_Name_Clean"] == site].copy()
  daily_ts = site_df.groupby("Date Local")["NO2_ppb"].mean().reset_index()
  daily_ts = daily_ts.sort_values("Date Local")

  # THE NEMESIS LOGIC: Predict today using yesterday (lag = 1 day)
  daily_ts["Persistent_Pred"] = daily_ts["NO2_ppb"].shift(1)

  # Drop Day 0 since it has no historical 'yesterday' to reference
  eval_df = daily_ts.dropna(subset=["Persistent_Pred", "NO2_ppb"]).copy()

  actual = eval_df["NO2_ppb"].values
  predicted = eval_df["Persistent_Pred"].values
  n_days = len(actual)

  # Calculate Regression Metrics
  rmse = np.sqrt(mean_squared_error(actual, predicted))
  mae = mean_absolute_error(actual, predicted)
  r2 = r2_score(actual, predicted)
  metrics_summary[site] = {"RMSE": rmse, "MAE": mae, "R2": r2, "N": n_days}

  print(
      f"   -> {site:14s} | RMSE: {rmse:5.2f} ppb | MAE: {mae:5.2f} ppb | R²:"
      f" {r2:6.3f}"
  )

  max_val = np.ceil(max(np.max(actual), np.max(predicted)) * 1.1)

  # Plot 1:1 Perfection Line
  ax.plot(
      [0, max_val],
      [0, max_val],
      color="black",
      linestyle="--",
      linewidth=1.5,
      alpha=0.7,
      label="1:1 Line (Perfection)",
      zorder=1,
  )

  # Scatterplot of Actual vs Persistent Guess
  ax.scatter(
      predicted,
      actual,
      color=colors[site],
      alpha=0.5,
      s=45,
      edgecolor="black",
      linewidth=0.5,
      zorder=2,
      label="Daily Forecast Pair",
  )

  # Formatting Subplot
  ax.set_xlim(0, max_val)
  ax.set_ylim(0, max_val)
  ax.set_xlabel("Predicted NO₂ (Yesterday's Value) [ppb]", fontsize=11, fontweight="bold")
  if idx == 0:
    ax.set_ylabel("Actual Observed NO₂ [ppb]", fontsize=11, fontweight="bold")
  ax.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=0)
  ax.set_title(f"📍 {site}\nPersistent Forecast ($t-1$)", fontsize=13, fontweight="bold")
  ax.legend(loc="upper left", fontsize=9, frameon=True, facecolor="white")

  # Statistics Overlay Box (Bottom Right)
  stats_box = (
      "Persistent Model\n"
      "Formula: $\\hat{y}_t = y_{t-1}$\n\n"
      "Performance:\n"
      f"RMSE: {rmse:.2f} ppb\n"
      f"MAE:  {mae:.2f} ppb\n"
      f"R²:   {r2:.3f}\n\n"
      f"N = {n_days} days"
  )
  ax.text(
      0.95,
      0.05,
      stats_box,
      transform=ax.transAxes,
      fontsize=8.5,
      verticalalignment="bottom",
      horizontalalignment="right",
      fontfamily="monospace",
      bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="gray", alpha=0.95),
      zorder=3,
  )

fig_scat.suptitle(
    "Actual vs. Predicted NO₂ - Persistent Model ('The Nemesis')\nLag-1 Daily"
    " Autocorrelation across Los Angeles Basin Sites (2023–2024)",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

out_scat = OUTPUT_DIR / "persistent_model_scatterplots.png"
plt.savefig(out_scat, dpi=300, bbox_inches="tight")
plt.close(fig_scat)
print(f"   ✅ Saved Scatterplot Figure to: {out_scat}")
print("=" * 75)


# ==============================================================================
# 5. GENERATE TIME SERIES & RESIDUAL PLOTS PER SITE
# ==============================================================================
def plot_persistent_timeseries_and_residuals(site_name, color):
  print(f"🎨 Generating Time Series & Residuals for: {site_name}...")
  site_df = df_sites[df_sites["Site_Name_Clean"] == site_name].copy()
  daily_ts = site_df.groupby("Date Local")["NO2_ppb"].mean().reset_index()
  daily_ts = daily_ts.sort_values("Date Local")
  daily_ts["Persistent_Pred"] = daily_ts["NO2_ppb"].shift(1)
  eval_df = daily_ts.dropna(subset=["Persistent_Pred", "NO2_ppb"]).copy()

  dates = eval_df["Date Local"]
  actual = eval_df["NO2_ppb"].values
  predicted = eval_df["Persistent_Pred"].values
  residuals = predicted - actual  # Predicted - Actual (Slide 19 formula)

  rmse = metrics_summary[site_name]["RMSE"]
  mae = metrics_summary[site_name]["MAE"]
  r2 = metrics_summary[site_name]["R2"]
  res_std = np.std(residuals)

  # --------------------------------------------------------------------------
  # FIGURE 1: OBSERVED vs PERSISTENT TIME SERIES
  # --------------------------------------------------------------------------
  fig1, ax1 = plt.subplots(figsize=(15, 8))

  # Plot Shaded ±1 RMSE Error Band around the forecast
  ax1.fill_between(
      dates,
      predicted - rmse,
      predicted + rmse,
      color="#fff3cd",
      alpha=0.6,
      label=f"±1 RMSE Error Band (±{rmse:.2f} ppb)",
      zorder=1,
  )

  ax1.plot(
      dates,
      actual,
      color=color,
      linewidth=1.3,
      alpha=0.85,
      label="Observed True NO₂",
      zorder=2,
  )
  ax1.plot(
      dates,
      predicted,
      color="#333333",
      linestyle="--",
      linewidth=1.2,
      alpha=0.8,
      label="Persistent Forecast ($y_{t-1}$)",
      zorder=3,
  )

  ax1.set_ylabel("NO₂ Daily Mean (ppb)", fontsize=12, fontweight="bold")
  ax1.set_xlabel("Date", fontsize=12, fontweight="bold")
  ax1.set_ylim(0, max(actual) * 1.18)
  ax1.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=0)
  ax1.set_title(
      f"Persistent Model Forecast vs Observed NO₂ - {site_name} Site"
      " (2023–2024)",
      fontsize=15,
      fontweight="bold",
      pad=15,
  )
  ax1.legend(
      loc="upper right", frameon=True, facecolor="white", edgecolor="gray", fontsize=10.5
  )

  perf_box1 = (
      "Persistent Forecast ($t-1$)\n"
      "Formula: $\\hat{y}_t = y_{t-1}$\n\n"
      "Overall Performance:\n"
      f"  RMSE: {rmse:.2f} ppb\n"
      f"  MAE:  {mae:.2f} ppb\n"
      f"  R²:   {r2:.3f}\n\n"
      f"N = {len(actual)} days"
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

  ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
  ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
  plt.xticks(fontsize=10, fontweight="bold")

  clean_site = site_name.lower().replace(" ", "_")
  out_file1 = OUTPUT_DIR / f"timeseries_observed_vs_persistent_{clean_site}.png"
  plt.savefig(out_file1, dpi=300, bbox_inches="tight")
  plt.close(fig1)

  # --------------------------------------------------------------------------
  # FIGURE 2: TIME SERIES OF RESIDUALS
  # --------------------------------------------------------------------------
  fig2, ax2 = plt.subplots(figsize=(15, 8))

  ax2.fill_between(
      dates,
      -res_std,
      res_std,
      color="#d1e7dd",
      alpha=0.6,
      label=f"±1 Residual Std Dev (±{res_std:.2f} ppb)",
      zorder=1,
  )
  ax2.axhline(
      0,
      color="#333333",
      linestyle="-",
      linewidth=1.8,
      label="Zero Line (Perfect Forecast)",
      zorder=2,
  )
  ax2.plot(
      dates,
      residuals,
      color="#0f5132",
      linewidth=1.1,
      alpha=0.9,
      label="Residuals (Predicted - Actual)",
      zorder=3,
  )

  ax2.set_ylabel(
      "Residual (Predicted - Actual) [ppb]", fontsize=12, fontweight="bold"
  )
  ax2.set_xlabel("Date", fontsize=12, fontweight="bold")
  ax2.set_ylim(min(residuals) * 1.2, max(residuals) * 1.3)
  ax2.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=0)
  ax2.set_title(
      f"Persistent Model Residuals - {site_name} Site (2023–2024)\n(Day-to-Day"
      " Forecast Error)",
      fontsize=15,
      fontweight="bold",
      pad=15,
  )
  ax2.legend(
      loc="upper right", frameon=True, facecolor="white", edgecolor="gray", fontsize=10.5
  )

  perf_box2 = (
      f"Persistent Residuals ({site_name})\n\n"
      f"Residual mean: {np.mean(residuals):.4f} ppb\n"
      f"Residual std:  {res_std:.2f} ppb\n\n"
      "Performance:\n"
      f"  RMSE: {rmse:.2f} ppb\n"
      f"  MAE:  {mae:.2f} ppb\n"
      f"  R²:   {r2:.3f}\n\n"
      "Residual range:\n"
      f"  Min: {np.min(residuals):6.2f} ppb\n"
      f"  Max: {np.max(residuals):6.2f} ppb"
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

  ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
  ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
  plt.xticks(fontsize=10, fontweight="bold")

  out_file2 = OUTPUT_DIR / f"residuals_persistent_model_{clean_site}.png"
  plt.savefig(out_file2, dpi=300, bbox_inches="tight")
  plt.close(fig2)
  print(f"   ✅ Saved Time Series & Residual plots for {site_name}!")


for site in target_sites:
  plot_persistent_timeseries_and_residuals(site, colors[site])

print("\n🎯 ALL PERSISTENT MODEL GENERATIONS SUCCESSFULLY FINISHED!")