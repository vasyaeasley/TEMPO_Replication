import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. SETUP & PATH CONFIGURATION
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "epa_from_internet_daily"
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

print("🌍 STARTING PHASE 1: SENSOR CLUSTER VERIFICATION 🌍")
print("=" * 70)

# ==============================================================================
# 2. LOAD MONITORING SITE COORDINATES
# ==============================================================================
# Load 2024 EPA daily file to extract unique monitoring stations
no2_file = RAW_DIR / "daily_42602_2024.csv"
if not no2_file.exists():
  # Fallback path check
  no2_file = BASE_DIR / "raw" / "epa_from_internet_daily" / "daily_42602_2024.csv"

print(f"⏳ Extracting unique monitoring sites from {no2_file.name}...")
df_raw = pd.read_csv(
    no2_file,
    usecols=[
        "State Code",
        "County Code",
        "Site Num",
        "Latitude",
        "Longitude",
        "Local Site Name",
    ],
    low_memory=False,
)

# Filter for California sites (State Code == 6)
df_ca = df_raw[df_raw["State Code"] == 6].copy()

# Create unique Site ID and extract unique station locations
df_ca["Site_ID"] = (
    df_ca["State Code"].astype(str).str.zfill(2)
    + "-"
    + df_ca["County Code"].astype(str).str.zfill(3)
    + "-"
    + df_ca["Site Num"].astype(str).str.zfill(4)
)

sites_df = (
    df_ca.groupby("Site_ID")
    .agg({
        "Latitude": "first",
        "Longitude": "first",
        "Local Site Name": "first",
    })
    .reset_index()
)

print(f"✅ Found {len(sites_df)} unique California monitoring sites in dataset.")

# ==============================================================================
# 3. CHECK FOR EXISTING CLUSTERS OR RUN K-MEANS (5 CLUSTERS)
# ==============================================================================
if "Cluster" in sites_df.columns:
  print("✅ An existing 'Cluster' column was found in the dataframe!")
else:
  print(
      "ℹ️ No existing cluster column found. Applying K-Means (n_clusters=5) on"
      " spatial coordinates..."
  )
  # Fit K-Means on Latitude and Longitude to replicate paper's 5 proximity clusters
  kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
  sites_df["Cluster"] = kmeans.fit_predict(
      sites_df[["Latitude", "Longitude"]]
  )
  sites_df["Cluster"] = sites_df["Cluster"] + 1  # Label as Cluster 1 to 5

# ==============================================================================
# 4. EXECUTE SPATIALLY AWARE 60/40 TRAIN/TEST SPLIT
# ==============================================================================
print("\n📊 Executing 60% Train / 40% Test split per cluster...")
train_sites = []
test_sites = []

for cluster_num, group in sites_df.groupby("Cluster"):
  # Split 60/40 within each individual cluster
  train_group, test_group = train_test_split(
      group, test_size=0.40, random_state=42
  )
  train_sites.append(train_group)
  test_sites.append(test_group)
  print(
      f"   -> Cluster {cluster_num}: Total={len(group)} | Train={len(train_group)}"
      f" (60%) | Test={len(test_group)} (40%)"
  )

df_train = pd.concat(train_sites)
df_test = pd.concat(test_sites)
sites_df["Split"] = np.where(sites_df["Site_ID"].isin(df_train["Site_ID"]), "Train", "Test")

print("-" * 70)
print(
    f"🏆 TOTALS: Train Sites = {len(df_train)} | Test Sites = {len(df_test)}"
)
print("=" * 70)

# ==============================================================================
# 5. GENERATE VERIFICATION MAP (MIRRORS FIGURE 1 & 2 IN PAPER)
# ==============================================================================
print("🎨 Generating visual cluster verification map...")
fig, ax = plt.subplots(figsize=(12, 10))

cluster_colors = {
    1: "#1f77b4",  # Blue
    2: "#ff7f0e",  # Orange
    3: "#2ecc71",  # Green
    4: "#d62728",  # Red
    5: "#9467bd",  # Purple
}

for cluster_num, group in sites_df.groupby("Cluster"):
  # Plot training sites as dots
  train_sub = group[group["Split"] == "Train"]
  ax.scatter(
      train_sub["Longitude"],
      train_sub["Latitude"],
      color=cluster_colors[cluster_num],
      s=70,
      marker="o",
      edgecolor="black",
      linewidth=0.8,
      label=f"Cluster {cluster_num} (Train)",
      zorder=3,
  )

  # Plot testing sites as triangles to distinguish train vs test
  test_sub = group[group["Split"] == "Test"]
  ax.scatter(
      test_sub["Longitude"],
      test_sub["Latitude"],
      color=cluster_colors[cluster_num],
      s=80,
      marker="^",
      edgecolor="black",
      linewidth=0.8,
      label=f"Cluster {cluster_num} (Test)",
      zorder=4,
  )

ax.set_title(
    "EPA Surface Monitoring Sites - 5 Spatial K-Means Clusters\n60% Train (Circles) /"
    " 40% Test (Triangles) Spatially Aware Split",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Longitude", fontsize=11, fontweight="bold")
ax.set_ylabel("Latitude", fontsize=11, fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.5, color="gray", zorder=0)

# Legend formatting
handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles,
    labels,
    loc="upper right",
    frameon=True,
    facecolor="white",
    edgecolor="gray",
    ncol=2,
    fontsize=9.5,
)

out_file = OUTPUT_DIR / "verify_sensor_clusters_map.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"✅ Saved Verification Map to: {out_file}")
print("🎯 PHASE 1 CLUSTER VERIFICATION FINISHED!")