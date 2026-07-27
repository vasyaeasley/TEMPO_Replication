import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / 'data' / 'raw' / 'epa_summary' / 'annual_conc_by_monitor_2025.csv' # Adjust filename if needed

print("Loading EPA Annual Summary Data...")
df = pd.read_csv(csv_path)

# 2. Filter strictly for NO2 (Parameter Code 42602) and standard sample durations
no2_df = df[(df['Parameter Code'] == 42602) & (df['Sample Duration'] == '1 HOUR')].copy()

# Deduplicate so we only have one primary record per site
no2_df = no2_df.drop_duplicates(subset=['Latitude', 'Longitude'])

# Extract stats for the top-right text box
n_stations = len(no2_df)
median_max = no2_df['1st Max Value'].median()
mean_max = no2_df['1st Max Value'].mean()

print(f"Filtered to {n_stations} NO2 stations. Plotting map...")

# 3. Load Country and State Boundaries from Natural Earth
print("Loading boundaries from Natural Earth...")

# Load World Countries (provides Canada, Mexico, and background landmasses)
world_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(world_url)

# Load State/Province Boundaries (provides internal U.S. state lines)
states_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_1_states_provinces.zip"
states = gpd.read_file(states_url)

# Safely filter for US states regardless of column casing
admin_col = next((col for col in states.columns if col.lower() == 'admin'), 'admin')
us_states = states[states[admin_col] == "United States of America"]

# 4. Plotting the Map Layers
fig, ax = plt.subplots(figsize=(15, 8))

# Background grid
ax.grid(True, linestyle='--', alpha=0.5, color='gray', zorder=1)

# Layer 1: World background (draws Canada & Mexico in a subtle light gray)
world.plot(ax=ax, color='#eaeaea', edgecolor='black', linewidth=1.0, zorder=2)

# Layer 2: U.S. States (highlights the CONUS in white with thin state borders)
us_states.plot(ax=ax, color='whitesmoke', edgecolor='black', linewidth=0.6, zorder=3)

# Set map limits strictly to Continental U.S. (CONUS)
ax.set_xlim([-128, -65])
ax.set_ylim([24, 51])

# --- THE MISSING LINE! Define marker sizes proportional to 1st Max Value ---
sizes = no2_df['1st Max Value'] * 2.5

# Plot the Stations on top of everything (zorder=4)
scatter = ax.scatter(
    no2_df['Longitude'], 
    no2_df['Latitude'], 
    s=sizes, 
    color='dodgerblue', 
    alpha=0.7, 
    edgecolor='white', 
    linewidth=0.5,
    zorder=4
)

# 5. Add Title and Formatting to match PNG
plt.title(
    "EPA NO$_2$ Monitoring Stations - 2025\nMarker size proportional to maximum NO$_2$ value", 
    fontsize=14, fontweight='bold', pad=15
)
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# 6. Add Top-Right Statistics Box
stats_text = f"n = {n_stations} stations\nMedian max: {median_max:.1f} ppb\nMean max: {mean_max:.1f} ppb"
ax.text(
    0.98, 0.95, stats_text, 
    transform=ax.transAxes, 
    fontsize=10, 
    verticalalignment='top', 
    horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
)

# 7. Add Bottom-Left Legend Box for Marker Sizes
# Create dummy plots for the legend
for sample_val in [3, 42, 97]:
    ax.scatter([], [], s=sample_val*2.5, color='dodgerblue', alpha=0.7, edgecolor='white', label=f"{sample_val} ppb")

legend = ax.legend(
    title="Max NO$_2$ Value", 
    loc="lower left", 
    frameon=True, 
    facecolor='white', 
    edgecolor='gray',
    labelspacing=1.2,
    borderpad=0.8
)
legend.get_title().set_fontweight('bold')

# Save exactly like his PNG
output_png = BASE_DIR / 'models' / 'epa_stations_2025_replicated.png'
plt.tight_layout()
plt.savefig(output_png, dpi=300)
print(f"Success! Graph saved to {output_png}")