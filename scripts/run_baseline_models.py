import os
from pathlib import Path
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

print("🤖 STARTING DUMB BASELINE MODELING (MEAN, MEDIAN, MODE) 🤖")
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
# 4. BASELINE EVALUATION ENGINE & PLOTTING
# ==============================================================================
def evaluate_and_plot_baselines(site_name, data_series, color="#1f77b4"):
  print(f"\n📍 Running Baseline Models for: {site_name}")

  actual = data_series.values
  n_days = len(actual)

  # 1. Calculate the 3 "Dumb" Single-Value Guesses
  val_mean = np.mean(actual)
  val_median = np.median(actual)

  # Calculate Mode using 0.5 ppb binned data (same logic as histograms)
  binned_data = np.round(actual * 2) / 2
  mode_res = stats.mode(binned_data, keepdims=True)
  val_mode = mode_res.mode[0] if len(mode_res.mode) > 0 else val_median

  models = {
      "Mean Model": val_mean,
      "Median Model": val_median,
      "Mode Model": val_mode,
  }

  # Prepare 1x3 Subplot Grid mirroring Slide 18
  fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
  fig.subplots_adjust(wspace=0.15)

  max_axis_val = np.ceil(np.max(actual) * 1.1)

  for idx, (m_name, pred_val) in enumerate(models.items()):
    ax = axes[idx]

    # Create an array where every single day is predicted as the same constant number
    predicted = np.full_like(actual, fill_value=pred_val)

    # Calculate Regression Metrics
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    print(
        f"   -> {m_name:12s} | Guess: {pred_val:5.2f} ppb | RMSE: {rmse:5.2f} |"
        f" MAE: {mae:5.2f} | R²: {r2:6.3f}"
    )

    # Plot the 1:1 Diagonal Perfection Line
    ax.plot(
        [0, max_axis_val],
        [0, max_axis_val],
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="1:1 Line (Perfection)",
        zorder=1,
    )

    # Plot the Horizontal Line indicating the constant guess level
    ax.axhline(
        pred_val,
        color="#e74c3c",
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
        label=f"Constant Guess = {pred_val:.2f} ppb",
        zorder=2,
    )

    # Scatterplot of Actual vs Predicted (Forms the vertical pillar!)
    ax.scatter(
        predicted,
        actual,
        color=color,
        alpha=0.45,
        s=40,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )

    # Formatting Subplot
    ax.set_xlim(0, max_axis_val)
    ax.set_ylim(0, max_axis_val)
    ax.set_xlabel(f"Predicted NO₂ ({m_name}) [ppb]", fontsize=11, fontweight="bold")
    if idx == 0:
      ax.set_ylabel("Actual NO₂ [ppb]", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=0)
    ax.set_title(f"{m_name}\n(Guess: {pred_val:.2f} ppb)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor="white")

    # Statistics & Performance Overlay Box (Bottom Right)
    stats_box = (
        f"{m_name}\n\n"
        f"Prediction: {pred_val:.2f} ppb\n"
        "(constant for all days)\n\n"
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
        zorder=4,
    )

  fig.suptitle(
      f"Actual vs. Predicted NO₂ - Baseline ('Dumb') Models\nSite: {site_name} (2023–2024)",
      fontsize=16,
      fontweight="bold",
      y=0.98,
  )

  clean_filename = site_name.lower().replace(" ", "_")
  out_file = OUTPUT_DIR / f"baseline_models_scatterplot_{clean_filename}.png"
  plt.savefig(out_file, dpi=300, bbox_inches="tight")
  plt.close(fig)
  print(f"   ✅ Saved Diagnostic Figure to: {out_file}")


# ==============================================================================
# 5. EXECUTE ACROSS ALL TARGET SITES
# ==============================================================================
colors = {
    "Compton": "#d62728",  # Deep Red
    "Pomona": "#ff7f0e",  # Orange
    "Santa Clarita": "#1f77b4",  # Royal Blue
}

for site in target_sites:
  site_series = df_sites[df_sites["Site_Name_Clean"] == site]["NO2_ppb"].dropna()
  evaluate_and_plot_baselines(site, site_series, color=colors[site])

print("\n🎯 ALL BASELINE MODEL EVALUATIONS SUCCESSFULLY FINISHED!")