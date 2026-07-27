from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
EPA_DIR_CANDIDATES = [
    RAW_DIR / "epa_from_internet",
    RAW_DIR / "epa_from_internet_hourly",
]
EPA_DIR = next((p for p in EPA_DIR_CANDIDATES if p.exists()), EPA_DIR_CANDIDATES[0])
OUT_DIR = BASE_DIR / "data" / "processed" / "airnow_station_analysis"

YEARS = list(range(2020, 2026))
PARAMETERS = ["NO2", "TEMP", "WIND", "PRESS", "RH_DP"]
FILE_PRIORITY = {
    "NO2": ["hourly_42602_{year}.csv"],
    "TEMP": ["hourly_TEMP_{year}.csv", "daily_TEMP_{year}.csv"],
    "WIND": ["hourly_WIND_{year}.csv", "daily_WIND_{year}.csv"],
    "PRESS": ["hourly_PRESS_{year}.csv", "daily_PRESS_{year}.csv"],
    "RH_DP": ["hourly_RH_DP_{year}.csv", "daily_RH_DP_{year}.csv"],
}


def parse_int(token: str) -> Optional[int]:
    t = str(token).strip().strip('"')
    if not t:
        return None
    try:
        return int(float(t))
    except Exception:
        return None


def find_file(param: str, year: int) -> Optional[Path]:
    for patt in FILE_PRIORITY[param]:
        fp = EPA_DIR / patt.format(year=year)
        if fp.exists():
            return fp
    return None


def extract_ids(fp: Path) -> Tuple[Set[Tuple[int, int, int]], int]:
    ids: Set[Tuple[int, int, int]] = set()
    bad = 0
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        next(f, None)
        for line in f:
            parts = line.split(",", 6)
            if len(parts) < 3:
                bad += 1
                continue
            sc = parse_int(parts[0])
            cc = parse_int(parts[1])
            sn = parse_int(parts[2])
            if sc is None or cc is None or sn is None:
                continue
            ids.add((sc, cc, sn))
    return ids, bad


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_year: Dict[int, Dict[str, Set[Tuple[int, int, int]]]] = {y: {} for y in YEARS}
    union_by_param: Dict[str, Set[Tuple[int, int, int]]] = {p: set() for p in PARAMETERS}
    files_used: Dict[str, Dict[int, str]] = {p: {} for p in PARAMETERS}
    total_bad = 0

    for y in YEARS:
        for p in PARAMETERS:
            fp = find_file(p, y)
            if fp is None:
                per_year[y][p] = set()
                continue
            files_used[p][y] = fp.name
            ids, bad = extract_ids(fp)
            total_bad += bad
            per_year[y][p] = ids
            union_by_param[p].update(ids)

    counts_rows = []
    all5_rows = []
    for y in YEARS:
        row = {"year": y}
        for p in PARAMETERS:
            row[p] = len(per_year[y].get(p, set()))
        counts_rows.append(row)

        yr_sets = [per_year[y].get(p, set()) for p in PARAMETERS]
        all5_rows.append(
            {
                "year": y,
                "stations_all_5_parameters": len(set.intersection(*yr_sets)) if all(yr_sets) else 0,
            }
        )

    counts_df = pd.DataFrame(counts_rows)
    all5_df = pd.DataFrame(all5_rows)

    counts_df.to_csv(OUT_DIR / "station_counts_by_year.csv", index=False)
    all5_df.to_csv(OUT_DIR / "full_coverage_station_counts_by_year.csv", index=False)

    summary = [
        "AirNow/EPA station analysis summary",
        "=================================",
        "",
        "Reprocessed from newly added epa_from_internet files",
        "Primary monitor key: State Code + County Code + Site Num",
        f"Total malformed rows skipped while parsing: {total_bad}",
        "",
        "Union unique monitors by parameter (2020-2025):",
    ]
    for p in PARAMETERS:
        summary.append(f"- {p}: {len(union_by_param[p])}")

    summary.append("")
    summary.append("Overall unique monitors across all parameters:")
    summary.append(f"- {len(set.union(*union_by_param.values()))}")

    summary.append("")
    summary.append("Files used:")
    for p in PARAMETERS:
        summary.append(f"- {p}: {files_used[p] if files_used[p] else 'none'}")

    missing = []
    for y in YEARS:
        for p in PARAMETERS:
            if y not in files_used[p]:
                missing.append(f"{p} file missing for {y}")
    summary.append("")
    summary.append("Missing data flags:")
    summary.extend([f"- {m}" for m in missing] if missing else ["- None"])

    (OUT_DIR / "data_availability_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("Wrote:")
    print(" - station_counts_by_year.csv")
    print(" - full_coverage_station_counts_by_year.csv")
    print(" - data_availability_summary.txt")


if __name__ == "__main__":
    main()
