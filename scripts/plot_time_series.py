import os
from pathlib import Path
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "epa_from_internet_daily"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

print("📈 STARTING 2023-2024 NO2 TIME SERIES GENERATION (FIXED STATS BOX) 📈")
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
# 3. FILTER FOR SELECTED SITES & FIND GLOBAL MAXIMUM
# ==============================================================================
target_sites = ["Compton", "Pomona", "Santa Clarita"]
df_all["Site_Name_Clean"] = df_all["Local Site Name"].astype(str).str.strip()
df_sites = df_all[df_all["Site_Name_Clean"].isin(target_sites)].copy()

# Calculate the global maximum across all 3 sites to dictate the shared y-axis ceiling
global_max_ppb = df_sites["NO2_ppb"].max()
print(f"✅ Extracted data for: {target_sites}")
print(
    f"🌐 Global Maximum NO2 Peak across all sites: {global_max_ppb:.1f} ppb"
    " (Dictates Shared Y-Axis)"
)
print("=" * 70)


# ==============================================================================
# 4. PLOTTING ENGINE (GENERATES BOTH ZOOMED AND SHARED AXIS MAPS)
# ==============================================================================
def generate_time_series_plot(shared_axis=False, output_filename=""):
  mode_label = "SHARED Y-AXIS" if shared_axis else "ZOOMED INDIVIDUAL AXES"
  print(f"🎨 Generating Plot Mode: {mode_label}...")

  fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
  fig.subplots_adjust(hspace=0.18, bottom=0.09, top=0.92)

  colors = {
      "Compton": "#d62728",  # Deep Red (Urban Core)
      "Pomona": "#ff7f0e",  # Orange (Eastern Valley)
      "Santa Clarita": "#1f77b4",  # Royal Blue (Northern Suburban Basin)
  }

  legend_handles = []
  legend_labels = []

  for i, site_name in enumerate(target_sites):
    ax = axes[i]
    site_data = df_sites[df_sites["Site_Name_Clean"] == site_name].copy()
    daily_ts = site_data.groupby("Date Local")["NO2_ppb"].mean().reset_index()
    daily_ts["Rolling_7D"] = (
        daily_ts["NO2_ppb"].rolling(window=7, min_periods=1).mean()
    )

    # Calculate Site Statistics
    mean_val = daily_ts["NO2_ppb"].mean()
    median_val = daily_ts["NO2_ppb"].median()
    min_val = daily_ts["NO2_ppb"].min()
    max_val = daily_ts["NO2_ppb"].max()
    range_val = max_val - min_val

    # Plot raw daily data
    (l1,) = ax.plot(
        daily_ts["Date Local"],
        daily_ts["NO2_ppb"],
        color=colors[site_name],
        alpha=0.25,
        linewidth=0.8,
        label="Daily Mean",
    )

    # Plot 7-Day Rolling Average
    (l2,) = ax.plot(
        daily_ts["Date Local"],
        daily_ts["Rolling_7D"],
        color=colors[site_name],
        alpha=0.95,
        linewidth=2.2,
        label="7-Day Rolling Mean",
    )

    # Plot bold Mean line
    l3 = ax.axhline(
        mean_val,
        color="black",
        linestyle="--",
        linewidth=1.8,
        alpha=0.85,
        label="Site Mean",
    )

    # Plot dotted Median line
    l4 = ax.axhline(
        median_val,
        color="#555555",
        linestyle=":",
        linewidth=1.6,
        alpha=0.85,
        label="Site Median",
    )

    if i == 0:
      legend_handles = [l1, l2, l3, l4]
      legend_labels = [
          "Daily Mean (Raw)",
          "7-Day Rolling Mean",
          "Overall Mean (Dashed)",
          "Overall Median (Dotted)",
      ]

    # Y-AXIS LOGIC: Bunted up slightly to 1.22 to clear the newly raised stats box
    if shared_axis:
      ax.set_ylim(0, global_max_ppb * 1.22)
    else:
      ax.set_ylim(0, max_val * 1.22)

    ax.set_ylabel("NO₂ (ppb)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, color="gray")

    # Site Name Badge (Kept in place)
    ax.text(
        0.015,
        0.85,
        f"📍 {site_name}",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=colors[site_name],
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=colors[site_name],
            alpha=0.9,
        ),
    )

    # Statistics Box (Raised from y=0.85 to y=0.96 and compacted padding)
    stats_text = (
        f"Mean:   {mean_val:4.1f} ppb\n"
        f"Median: {median_val:4.1f} ppb\n"
        f"Range:  {range_val:4.1f} ppb\n"
        f"(Min: {min_val:.1f} | Max: {max_val:.1f})"
    )
    ax.text(
        0.985,
        0.96,  # Raised up near the very ceiling of the plot box
        stats_text,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        horizontalalignment="right",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="square,pad=0.35",  # Compacted padding slightly
            facecolor="white",
            edgecolor="gray",
            alpha=0.9,
        ),
    )

  # Master Bottom Legend
  fig.legend(
      legend_handles,
      legend_labels,
      loc="lower center",
      bbox_to_anchor=(0.5, 0.005),
      ncol=4,
      frameon=True,
      facecolor="white",
      edgecolor="gray",
      fontsize=10.5,
  )

  # Format X-Axis Dates
  axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
  axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
  plt.xticks(rotation=0, fontsize=10.5, fontweight="bold")

  title_suffix = (
      "(Shared Y-Axis Scale)" if shared_axis else "(Individual Zoomed Axes)"
  )
  fig.suptitle(
      "EPA Surface NO₂ Time Series (2023–2024) - Los Angeles Basin Typological"
      f" Comparison\n{title_suffix}",
      fontsize=16,
      fontweight="bold",
      y=0.97,
  )

  out_file = OUTPUT_DIR / output_filename
  plt.savefig(out_file, dpi=300, bbox_inches="tight")
  print(f"   -> Saved to: {out_file}")


# ==============================================================================
# 5. EXECUTE BOTH GENERATIONS
# ==============================================================================
# 1. Generate Original Zoomed Version
generate_time_series_plot(
    shared_axis=False, output_filename="la_basin_time_series_comparison.png"
)

# 2. Generate New Shared-Axis Version
generate_time_series_plot(
    shared_axis=True,
    output_filename="la_basin_time_series_comparison_shared_axis.png",
)

print("\n🎯 BOTH TIME SERIES PLOTS SUCCESSFULLY GENERATED AND SAVED!")