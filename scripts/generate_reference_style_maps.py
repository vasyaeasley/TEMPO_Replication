from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
EPA_DIR_CANDIDATES = [
    RAW_DIR / "epa_from_internet",
    RAW_DIR / "epa_from_internet_hourly",
]
EPA_DIR = next((p for p in EPA_DIR_CANDIDATES if p.exists()), EPA_DIR_CANDIDATES[0])
OUT_DIR = BASE_DIR / "data" / "processed" / "airnow_station_analysis"

YEAR = 2025

FILE_MAP = {
    "NO2": f"hourly_42602_{YEAR}.csv",
    "TEMP": f"hourly_TEMP_{YEAR}.csv",
    "PRESS": f"hourly_PRESS_{YEAR}.csv",
    "RH_DP": f"hourly_RH_DP_{YEAR}.csv",
    "WIND": f"hourly_WIND_{YEAR}.csv",
}


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


def read_site_ids_and_coords(path: Path) -> Tuple[Set[str], Dict[str, Tuple[float, float]]]:
    ids: Set[str] = set()
    coords: Dict[str, Tuple[float, float]] = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        next(f, None)
        for line in f:
            # We only need the first 7 columns for ID and coordinates.
            parts = line.split(",", 7)
            if len(parts) < 7:
                continue

            sc = parse_int(parts[0])
            cc = parse_int(parts[1])
            sn = parse_int(parts[2])
            lat = parse_float(parts[5])
            lon = parse_float(parts[6])
            if sc is None or cc is None or sn is None:
                continue

            sid = f"{sc:02d}-{cc:03d}-{sn:04d}"
            ids.add(sid)
            if lat is not None and lon is not None and sid not in coords:
                coords[sid] = (lat, lon)

    return ids, coords


def collect_monitor_sets() -> Tuple[Dict[str, Set[str]], Dict[str, Tuple[float, float]]]:
    sets: Dict[str, Set[str]] = {}
    coord_registry: Dict[str, Tuple[float, float]] = {}

    for param, filename in FILE_MAP.items():
        fp = EPA_DIR / filename
        if not fp.exists():
            raise FileNotFoundError(f"Missing required file: {fp}")
        ids, coords = read_site_ids_and_coords(fp)
        sets[param] = ids
        coord_registry.update(coords)

    return sets, coord_registry


def coords_from_ids(ids: Set[str], coord_registry: Dict[str, Tuple[float, float]]) -> np.ndarray:
    pts = [coord_registry[sid] for sid in sorted(ids) if sid in coord_registry]
    if not pts:
        return np.empty((0, 2), dtype=float)
    return np.array(pts, dtype=float)


def setup_us_axes(title: str):
    fig = plt.figure(figsize=(16, 8), facecolor="#d9d9d9")
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_facecolor("#d9d9d9")
    ax.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="black", linewidth=1.8, facecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor="black", linewidth=1.5)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="black", linewidth=1.0)

    gl = ax.gridlines(draw_labels=False, linewidth=0.6, color="gray", alpha=0.25, linestyle="-")
    gl.xlocator = plt.MultipleLocator(5)
    gl.ylocator = plt.MultipleLocator(5)

    ax.set_title(title, fontsize=22, fontweight="bold", pad=18)
    return fig, ax


def plot_no2_map(no2_points: np.ndarray, output_path: Path) -> None:
    fig, ax = setup_us_axes("EPA NO$_2$ Monitoring Stations - 2025")

    if len(no2_points) > 0:
        ax.scatter(
            no2_points[:, 1],
            no2_points[:, 0],
            s=95,
            marker="o",
            c="#1e88e5",
            alpha=0.75,
            edgecolors="white",
            linewidths=1.0,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

    ax.text(
        0.02,
        0.96,
        f"n = {len(no2_points)} stations",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=19,
        bbox={"facecolor": "#e6e6e6", "edgecolor": "#bfbfbf", "boxstyle": "round,pad=0.35", "alpha": 0.95},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_all5_map(all5_points: np.ndarray, output_path: Path) -> None:
    title = (
        "EPA Comprehensive Monitoring Sites - 2025\n"
        "(Sites measuring NO$_2$, Temperature, Pressure, Humidity/Dew Point, and Wind)"
    )
    fig, ax = setup_us_axes(title)

    if len(all5_points) > 0:
        ax.scatter(
            all5_points[:, 1],
            all5_points[:, 0],
            s=95,
            marker="D",
            c="#b71c1c",
            alpha=0.82,
            edgecolors="white",
            linewidths=1.0,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )

    ax.text(
        0.02,
        0.98,
        f"n = {len(all5_points)} comprehensive sites\nAll 5 parameters measured",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=17,
        bbox={"facecolor": "#e6e6e6", "edgecolor": "#bfbfbf", "boxstyle": "round,pad=0.35", "alpha": 0.95},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    monitor_sets, coord_registry = collect_monitor_sets()

    no2_ids = monitor_sets["NO2"]
    all5_ids = set.intersection(
        monitor_sets["NO2"],
        monitor_sets["TEMP"],
        monitor_sets["PRESS"],
        monitor_sets["RH_DP"],
        monitor_sets["WIND"],
    )

    no2_points = coords_from_ids(no2_ids, coord_registry)
    all5_points = coords_from_ids(all5_ids, coord_registry)

    no2_path = OUT_DIR / "epa_no2_stations_2025_reference_style.png"
    all5_path = OUT_DIR / "epa_comprehensive_sites_2025_reference_style.png"

    plot_no2_map(no2_points, no2_path)
    plot_all5_map(all5_points, all5_path)

    print(f"EPA source directory: {EPA_DIR}")
    print(f"NO2 station count: {len(no2_points)}")
    print(f"All-parameter station count: {len(all5_points)}")
    print(f"Saved: {no2_path}")
    print(f"Saved: {all5_path}")


if __name__ == "__main__":
    main()
