import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "epa_from_internet_daily"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

print("📊 STARTING 2023-2024 NO2 HISTOGRAM GENERATION 📊")
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
# 4. HELPER FUNCTION TO DRAW HISTOGRAM ON AN AXIS
# ==============================================================================
def draw_site_histogram(ax, site_name, data_series, color="#4185c4"):
  # 0.5 ppb Bins centered on the value
  min_val = np.floor(data_series.min())
  max_val = np.ceil(data_series.max())
  bins = np.arange(min_val - 0.25, max_val + 0.75, 0.5)

  # Calculate Statistics
  mean_val = data_series.mean()
  median_val = data_series.median()
  std_dev = data_series.std()
  min_stat = data_series.min()
  max_stat = data_series.max()
  n_days = len(data_series)

  # Calculate Mode using rounded 0.5 bins for meaningful peak detection
  binned_data = np.round(data_series * 2) / 2
  mode_res = stats.mode(binned_data, keepdims=True)
  mode_val = mode_res.mode[0] if len(mode_res.mode) > 0 else median_val

  # Plot Histogram
  ax.hist(
      data_series,
      bins=bins,
      color=color,
      edgecolor="white",
      linewidth=0.6,
      alpha=0.9,
  )

  # Plot Mean and Median Lines
  ax.axvline(
      median_val,
      color="#2ecc71",
      linestyle="--",
      linewidth=2.5,
      label=f"Median: {median_val:.2f} ppb",
      zorder=4,
  )
  ax.axvline(
      mean_val,
      color="#e74c3c",
      linestyle="--",
      linewidth=2.5,
      label=f"Mean: {mean_val:.2f} ppb",
      zorder=4,
  )

  # Formatting Axis
  ax.set_ylabel("Frequency (Days)", fontsize=11, fontweight="bold")
  ax.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=0)

  # Data Source Box (Top Left)
  source_text = (
      f"Data Source:\nEPA Air Quality System\nSite: {site_name}\nPeriod:"
      " 2023–2024"
  )
  ax.text(
      0.015,
      0.95,
      source_text,
      transform=ax.transAxes,
      fontsize=8.5,
      verticalalignment="top",
      bbox=dict(
          boxstyle="round,pad=0.5",
          facecolor="#f8f9fa",
          edgecolor="gray",
          alpha=0.9,
      ),
  )

  # Statistics Box (Top Right)
  stats_text = (
      "Statistics:\n"
      f"Mean:    {mean_val:6.2f} ppb\n"
      f"Median:  {median_val:6.2f} ppb\n"
      f"Mode:    {mode_val:6.2f} ppb\n"
      f"Std Dev: {std_dev:6.2f} ppb\n"
      f"Min:     {min_stat:6.2f} ppb\n"
      f"Max:     {max_stat:6.2f} ppb\n"
      f"N:       {n_days:6d} days"
  )
  ax.text(
      0.985,
      0.95,
      stats_text,
      transform=ax.transAxes,
      fontsize=9,
      verticalalignment="top",
      horizontalalignment="right",
      fontfamily="monospace",
      bbox=dict(
          boxstyle="round,pad=0.5",
          facecolor="#f8f9fa",
          edgecolor="gray",
          alpha=0.9,
      ),
  )

  # Legend Box (Middle Right)
  ax.legend(
      loc="center right",
      frameon=True,
      facecolor="#f8f9fa",
      edgecolor="gray",
      fontsize=10,
      shadow=True,
      borderpad=0.8,
  )


# ==============================================================================
# 5. GENERATE INDIVIDUAL HISTOGRAMS (MIRRORS SLIDE #10)
# ==============================================================================
colors = {
    "Compton": "#d62728",  # Deep Red
    "Pomona": "#ff7f0e",  # Orange
    "Santa Clarita": "#1f77b4",  # Royal Blue
}

for site_name in target_sites:
  print(f"🎨 Generating Individual Histogram for: {site_name}...")
  site_data = df_sites[df_sites["Site_Name_Clean"] == site_name][
      "NO2_ppb"
  ].dropna()

  fig, ax = plt.subplots(figsize=(14, 8))
  draw_site_histogram(ax, site_name, site_data, color=colors[site_name])

  ax.set_xlabel("NO₂ Daily Mean (ppb)", fontsize=12, fontweight="bold")
  ax.set_title(
      f"Distribution of NO₂ Daily Mean Concentrations - {site_name} Site"
      " (2023–2024)",
      fontsize=15,
      fontweight="bold",
      pad=15,
  )

  clean_filename = site_name.lower().replace(" ", "_")
  out_file = OUTPUT_DIR / f"histogram_no2_{clean_filename}.png"
  plt.savefig(out_file, dpi=300, bbox_inches="tight")
  plt.close(fig)
  print(f"   -> Saved to: {out_file}")

# ==============================================================================
# 6. GENERATE 3-PANEL VERTICAL COMPARISON FIGURE
# ==============================================================================
print("🎨 Generating 3-Panel Stacked Comparison Histogram...")
fig_comp, axes_comp = plt.subplots(3, 1, figsize=(15, 14), sharex=True)
fig_comp.subplots_adjust(hspace=0.22, top=0.93)

for i, site_name in enumerate(target_sites):
  ax = axes_comp[i]
  site_data = df_sites[df_sites["Site_Name_Clean"] == site_name][
      "NO2_ppb"
  ].dropna()
  draw_site_histogram(ax, site_name, site_data, color=colors[site_name])
  ax.set_title(
      f"📍 {site_name} Distribution", fontsize=13, fontweight="bold", loc="left"
  )

axes_comp[-1].set_xlabel("NO₂ Daily Mean (ppb)", fontsize=13, fontweight="bold")
fig_comp.suptitle(
    "EPA Surface NO₂ Concentration Distributions (2023–2024) - Los Angeles"
    " Basin Comparison",
    fontsize=16,
    fontweight="bold",
)

out_comp = OUTPUT_DIR / "la_basin_histograms_comparison.png"
plt.savefig(out_comp, dpi=300, bbox_inches="tight")
plt.close(fig_comp)
print(f"   -> Saved Comparison to: {out_comp}")

print("\n🎯 ALL HISTOGRAM GENERATIONS SUCCESSFULLY FINISHED!")