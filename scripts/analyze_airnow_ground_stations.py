from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BBox:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
EPA_DIR_CANDIDATES = [
    RAW_DIR / "epa_from_internet",
    RAW_DIR / "epa_from_internet_hourly",
]
EPA_DIR = next((p for p in EPA_DIR_CANDIDATES if p.exists()), EPA_DIR_CANDIDATES[0])
TEMPO_MONTHLY_DIR = BASE_DIR / "data" / "processed" / "tempo_monthly"
ROADS_GEOJSON = BASE_DIR / "data" / "raw" / "highway_map_california" / "National_Highway_System.geojson"
OUT_DIR = BASE_DIR / "data" / "processed" / "airnow_station_analysis"

YEARS = list(range(2020, 2026))
PARAMETERS = ["NO2", "TEMP", "WIND", "PRESS", "RH_DP"]
CSV_PARAM_FILE_PRIORITY = {
    "NO2": ["hourly_42602_{year}.csv"],
    "TEMP": ["hourly_TEMP_{year}.csv", "daily_TEMP_{year}.csv"],
    "WIND": ["hourly_WIND_{year}.csv", "daily_WIND_{year}.csv"],
    "PRESS": ["hourly_PRESS_{year}.csv", "daily_PRESS_{year}.csv"],
    "RH_DP": ["hourly_RH_DP_{year}.csv", "daily_RH_DP_{year}.csv"],
}

# Tighter LA Basin extent for the TEMPO overlay figure.
LA_BASIN = BBox(lon_min=-118.95, lon_max=-117.25, lat_min=33.55, lat_max=34.45)


def round_coord_key(lat: Iterable[float], lon: Iterable[float], ndp: int = 3) -> Set[Tuple[float, float]]:
    arr = np.column_stack([np.array(lat, dtype=float), np.array(lon, dtype=float)])
    arr = arr[np.isfinite(arr).all(axis=1)]
    arr = np.round(arr, ndp)
    return {tuple(x) for x in arr}


def parse_int(value: str) -> Optional[int]:
    txt = str(value).strip().strip('"')
    if not txt:
        return None
    try:
        return int(float(txt))
    except Exception:
        return None


def parse_float(value: str) -> Optional[float]:
    txt = str(value).strip().strip('"')
    if not txt:
        return None
    try:
        out = float(txt)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def find_best_file(param: str, year: int) -> Optional[Path]:
    for pattern in CSV_PARAM_FILE_PRIORITY[param]:
        candidate = EPA_DIR / pattern.format(year=year)
        if candidate.exists():
            return candidate
    return None


def parse_station_file(file_path: Path) -> Tuple[Set[str], Dict[str, Tuple[float, float]], int]:
    station_ids: Set[str] = set()
    station_coords: Dict[str, Tuple[float, float]] = {}
    bad_lines = 0

    # Read only the first 7 CSV fields by splitting the line. This is resilient to malformed
    # quotes further right in large hourly files and is enough for ID + coordinates.
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        next(f, None)
        for line in f:
            try:
                parts = line.split(",", 7)
                if len(parts) < 7:
                    bad_lines += 1
                    continue

                sc = parse_int(parts[0])
                cc = parse_int(parts[1])
                sn = parse_int(parts[2])
                lat = parse_float(parts[5])
                lon = parse_float(parts[6])
                if sc is None or cc is None or sn is None:
                    continue

                station_id = f"{sc:02d}-{cc:03d}-{sn:04d}"
                station_ids.add(station_id)
                if lat is not None and lon is not None and station_id not in station_coords:
                    station_coords[station_id] = (lat, lon)
            except Exception:
                bad_lines += 1
                continue

    return station_ids, station_coords, bad_lines


def load_station_sets_from_epa_files() -> Tuple[
    Dict[Tuple[str, int], Set[str]],
    Dict[str, Set[str]],
    pd.DataFrame,
    Dict[str, Dict[int, str]],
    int,
]:
    station_sets: Dict[Tuple[str, int], Set[str]] = {}
    records: List[dict] = []
    files_used: Dict[str, Dict[int, str]] = {p: {} for p in PARAMETERS}
    total_bad_lines = 0

    for year in YEARS:
        for param in PARAMETERS:
            src = find_best_file(param, year)
            if src is None:
                station_sets[(param, year)] = set()
                continue

            files_used[param][year] = src.name
            ids, coords, bad_lines = parse_station_file(src)
            total_bad_lines += bad_lines
            station_sets[(param, year)] = ids

            for sid, (lat, lon) in coords.items():
                records.append(
                    {
                        "station_id": sid,
                        "Latitude": lat,
                        "Longitude": lon,
                        "category": param,
                        "year": year,
                    }
                )

    all_points = pd.DataFrame.from_records(records).drop_duplicates() if records else pd.DataFrame()
    union_by_param_ids = {
        p: set().union(*(station_sets.get((p, y), set()) for y in YEARS)) for p in PARAMETERS
    }
    return station_sets, union_by_param_ids, all_points, files_used, total_bad_lines


def build_counts_table(station_sets: Dict[Tuple[str, int], Set[str]]) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        row = {"year": year}
        for param in PARAMETERS:
            row[param] = len(station_sets.get((param, year), set()))
        rows.append(row)
    return pd.DataFrame(rows)


def ids_to_coord_set(ids: Set[str], all_points: pd.DataFrame) -> Set[Tuple[float, float]]:
    if not ids or all_points.empty:
        return set()
    m = all_points[all_points["station_id"].isin(ids)][["Latitude", "Longitude"]].dropna().drop_duplicates()
    return round_coord_key(m["Latitude"], m["Longitude"]) if not m.empty else set()


def load_tempo_noon_layer() -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    tempo_files = sorted(TEMPO_MONTHLY_DIR.glob("*.nc"))
    if not tempo_files:
        raise FileNotFoundError(f"No TEMPO monthly files found in {TEMPO_MONTHLY_DIR}")

    tempo_path = tempo_files[-1]
    with nc.Dataset(tempo_path, "r") as ds:
        layer_stack = np.array(ds.variables["NO2_column"][:], dtype=float)
        lats = np.array(ds.variables["lat"][:], dtype=float)
        lons = np.array(ds.variables["lon"][:], dtype=float)
        time_var = ds.variables["time"]
        time_vals = np.array(time_var[:], dtype="int64")
        time_units = getattr(time_var, "units", "")

    match = re.match(r"nanoseconds since (.+)", time_units)
    if not match:
        raise ValueError(f"Unsupported TEMPO time units: {time_units}")

    origin = pd.Timestamp(match.group(1), tz="UTC")
    utc_times = origin + pd.to_timedelta(time_vals, unit="ns")
    local_times = utc_times.tz_convert("America/Los_Angeles")

    noon_indices = np.flatnonzero(local_times.hour == 12)
    if len(noon_indices) > 0:
        slice_idx = int(noon_indices[len(noon_indices) // 2])
    else:
        target_time = local_times[len(local_times) // 2].normalize() + pd.Timedelta(hours=12)
        slice_idx = int(np.argmin(np.abs(local_times - target_time)))

    layer = layer_stack[slice_idx, :, :] / 1e15
    layer = np.where(layer < 0, np.nan, layer)
    time_label = local_times[slice_idx].strftime("%Y-%m-%d %I:%M %p %Z")
    return layer, lats, lons, time_label


def make_us_station_map(all_points: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.scatter(
        all_points["Longitude"],
        all_points["Latitude"],
        s=8,
        alpha=0.35,
        c="tab:blue",
        edgecolors="none",
    )
    ax.set_xlim(-128, -65)
    ax.set_ylim(24, 50)
    ax.grid(ls=":", alpha=0.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Ground Station Locations from Available AirNow/EPA Data (2020-2025)")
    ax.text(
        0.99,
        0.01,
        f"Unique stations (coord-matched): {all_points[['lat_r', 'lon_r']].drop_duplicates().shape[0]:,}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def make_category_total_bar(union_sets: Dict[str, Set[str]], save_path: Path) -> None:
    order = ["NO2", "TEMP", "WIND", "PRESS", "RH_DP"]
    vals = [len(union_sets[p]) for p in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(order, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Unique stations")
    ax.set_title("Unique Ground Stations by Parameter Category (2020-2025, available data)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def make_category_yearly_lines(counts_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for p in PARAMETERS:
        ax.plot(counts_df["year"], counts_df[p], marker="o", lw=2, label=p)
    ax.set_xticks(YEARS)
    ax.set_ylabel("Stations with data in year")
    ax.set_xlabel("Year")
    ax.set_title("Stations by Parameter Over Time")
    ax.grid(ls=":", alpha=0.45)
    ax.legend(ncol=5, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def make_full_coverage_line(full_df: pd.DataFrame, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(full_df["year"], full_df["stations_all_5_parameters"], marker="o", lw=2.5, color="crimson")
    for _, r in full_df.iterrows():
        ax.text(r["year"], r["stations_all_5_parameters"], f"{int(r['stations_all_5_parameters'])}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(YEARS)
    ax.set_ylabel("Station count")
    ax.set_xlabel("Year")
    ax.grid(ls=":", alpha=0.5)
    ax.set_title("Stations Collecting All 5 Parameters (NO2, TEMP, WIND, PRESS, RH_DP)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def make_la_basin_map(
    all_station_coords: Set[Tuple[float, float]],
    full_station_coords: Set[Tuple[float, float]],
    save_path: Path,
) -> pd.DataFrame:
    tempo_layer, lats, lons, tempo_time_label = load_tempo_noon_layer()

    lat_mask = (lats >= LA_BASIN.lat_min) & (lats <= LA_BASIN.lat_max)
    lon_mask = (lons >= LA_BASIN.lon_min) & (lons <= LA_BASIN.lon_max)
    tempo_sub = tempo_layer[np.ix_(lat_mask, lon_mask)]
    lat_sub = lats[lat_mask]
    lon_sub = lons[lon_mask]

    all_df = pd.DataFrame(list(all_station_coords), columns=["lat", "lon"])
    all_df = all_df[
        (all_df["lat"].between(LA_BASIN.lat_min, LA_BASIN.lat_max))
        & (all_df["lon"].between(LA_BASIN.lon_min, LA_BASIN.lon_max))
    ].copy()

    full_df = pd.DataFrame(list(full_station_coords), columns=["lat", "lon"])
    full_df = full_df[
        (full_df["lat"].between(LA_BASIN.lat_min, LA_BASIN.lat_max))
        & (full_df["lon"].between(LA_BASIN.lon_min, LA_BASIN.lon_max))
    ].copy()

    roads = gpd.read_file(ROADS_GEOJSON)
    roads = roads.cx[LA_BASIN.lon_min : LA_BASIN.lon_max, LA_BASIN.lat_min : LA_BASIN.lat_max]

    fig, ax = plt.subplots(figsize=(10.5, 8))
    mesh = ax.pcolormesh(
        lon_sub,
        lat_sub,
        tempo_sub,
        cmap="inferno",
        shading="auto",
        vmin=np.nanpercentile(tempo_sub, 5),
        vmax=np.nanpercentile(tempo_sub, 95),
        zorder=1,
    )

    if len(roads) > 0:
        roads.plot(ax=ax, color="white", linewidth=0.7, alpha=0.75, zorder=2)

    ax.scatter(all_df["lon"], all_df["lat"], s=38, c="#00c8ff", alpha=1.0, edgecolors="black", linewidths=0.35, zorder=3, label=f"All stations in basin (n={len(all_df)})")
    if not full_df.empty:
        ax.scatter(full_df["lon"], full_df["lat"], s=95, marker="*", c="#7CFC00", edgecolors="black", linewidths=0.55, zorder=4, label=f"Stations with all 5 params (n={len(full_df)})")

    for lon in np.arange(np.floor(LA_BASIN.lon_min * 10) / 10, LA_BASIN.lon_max + 0.1, 0.1):
        ax.axvline(lon, color="white", alpha=0.25, lw=0.6, zorder=0)
    for lat in np.arange(np.floor(LA_BASIN.lat_min * 10) / 10, LA_BASIN.lat_max + 0.1, 0.1):
        ax.axhline(lat, color="white", alpha=0.25, lw=0.6, zorder=0)

    ax.set_xlim(LA_BASIN.lon_min, LA_BASIN.lon_max)
    ax.set_ylim(LA_BASIN.lat_min, LA_BASIN.lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"LA Basin Ground Stations with TEMPO Near-Noon NO$_2$, Roads, and 0.1° Grid\n{tempo_time_label}")
    ax.legend(loc="upper right", framealpha=0.9)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("TEMPO NO$_2$ Column (10$^{15}$ molec/cm$^2$)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)

    out = all_df.copy()
    out["collects_all_5_parameters"] = False
    if not full_df.empty:
        full_keys = set(zip(full_df["lat"], full_df["lon"]))
        out["collects_all_5_parameters"] = [tuple(x) in full_keys for x in zip(out["lat"], out["lon"])]
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    station_sets, union_sets, all_points, files_used, bad_lines = load_station_sets_from_epa_files()

    counts_df = build_counts_table(station_sets)

    full_rows = []
    for y in YEARS:
        by_year_sets = [station_sets.get((p, y), set()) for p in PARAMETERS]
        n_all = len(set.intersection(*by_year_sets)) if all(by_year_sets) else 0
        full_rows.append({"year": y, "stations_all_5_parameters": n_all})
    full_df = pd.DataFrame(full_rows)

    all_points = all_points.copy() if not all_points.empty else pd.DataFrame(columns=["station_id", "Latitude", "Longitude", "category", "year"])
    all_points["Latitude"] = pd.to_numeric(all_points["Latitude"], errors="coerce")
    all_points["Longitude"] = pd.to_numeric(all_points["Longitude"], errors="coerce")
    all_points = all_points.dropna(subset=["Latitude", "Longitude"]).copy()
    all_points["lat_r"] = all_points["Latitude"].round(3)
    all_points["lon_r"] = all_points["Longitude"].round(3)

    all_station_coords = set(zip(all_points["lat_r"], all_points["lon_r"]))
    full_station_ids_any_year = set.intersection(*(union_sets[p] for p in PARAMETERS)) if all(len(union_sets[p]) > 0 for p in PARAMETERS) else set()
    full_station_coords = ids_to_coord_set(full_station_ids_any_year, all_points)

    make_us_station_map(all_points, OUT_DIR / "us_ground_stations_map_2020_2025.png")
    make_category_total_bar(union_sets, OUT_DIR / "station_counts_by_category_total_2020_2025.png")
    make_category_yearly_lines(counts_df, OUT_DIR / "station_counts_by_category_yearly_2020_2025.png")
    make_full_coverage_line(full_df, OUT_DIR / "full_coverage_stations_yearly_2020_2025.png")
    la_df = make_la_basin_map(all_station_coords, full_station_coords, OUT_DIR / "la_basin_stations_elevation_roads_grid.png")

    counts_df.to_csv(OUT_DIR / "station_counts_by_year.csv", index=False)
    full_df.to_csv(OUT_DIR / "full_coverage_station_counts_by_year.csv", index=False)
    la_df.to_csv(OUT_DIR / "la_basin_station_inventory.csv", index=False)

    missing = []
    for y in YEARS:
        for p in PARAMETERS:
            if y not in files_used[p]:
                missing.append(f"{p} file missing for {y}")

    yearly_no2 = {int(r["year"]): int(r["NO2"]) for _, r in counts_df.iterrows()}
    no2_yearly_min = min(yearly_no2.values()) if yearly_no2 else 0
    no2_yearly_max = max(yearly_no2.values()) if yearly_no2 else 0

    summary_lines = [
        "AirNow/EPA station analysis summary",
        "=================================",
        "",
        "Parameter categories used: NO2, TEMP, WIND, PRESS, RH_DP",
        "Primary monitor key: State Code + County Code + Site Num",
        f"NO2 monitors per year span: {no2_yearly_min} to {no2_yearly_max}",
        f"NO2 monitor counts by year: {yearly_no2}",
        f"Total malformed rows skipped while parsing: {bad_lines}",
        "",
        "",
        "Missing data flags:",
    ]
    summary_lines.extend([f"- {m}" for m in missing] if missing else ["- None"])

    with open(OUT_DIR / "data_availability_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Saved outputs to: {OUT_DIR}")
    print("Generated files:")
    for fp in sorted(OUT_DIR.glob("*")):
        print(f" - {fp.name}")


if __name__ == "__main__":
    main()
