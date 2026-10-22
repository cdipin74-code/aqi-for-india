"""Compare Open-Meteo ERA5 against the NASA POWER research weather cache.

The live weather cache starts where the research dataset ends (2025-05-01),
so there are no common dates to compare directly. This script re-fetches
Open-Meteo for a window that lies INSIDE the research period (default
2025-01-01..2025-04-30), caches it separately under
data/raw/cpcb_live/weather_overlap_cache.json, and reports per-feature
correlation / bias / MAE against NASA POWER.

Nothing in data/interim or data/raw/aqi_source is modified.

Output: data/raw/cpcb_live/weather_overlap_validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fetch_weather_openmeteo import ARCHIVE_URL, fetch_window  # noqa: E402

INTERIM = ROOT / "data" / "interim"
LIVE_DIR = ROOT / "data" / "raw" / "cpcb_live"
NASA_WEATHER = INTERIM / "weather_daily.csv"
GEOCODE = INTERIM / "geocoded_locations.csv"
CACHE = LIVE_DIR / "weather_overlap_cache.json"
OUT = LIVE_DIR / "weather_overlap_validation.json"

FEATURES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_2m_mean",
    "wind_speed_2m_max",
    "surface_pressure_mean",
    "shortwave_radiation_sum",
]


def load_locations(cities: list[str]) -> list[dict]:
    geo = pd.read_csv(GEOCODE)
    out = []
    for city in cities:
        m = geo[geo["area"].str.strip().str.lower() == city.strip().lower()]
        if m.empty:
            print("WARN: no geocode for", city)
            continue
        r = m.iloc[0]
        out.append({
            "city": city,
            "location_key": r["location_key"],
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
        })
    return out


def fetch_cached(locations: list[dict], start: str, end: str) -> dict[str, list[dict]]:
    cache = json.loads(CACHE.read_text(encoding="utf8")) if CACHE.exists() else {}
    changed = False
    for loc in locations:
        key = loc["location_key"]
        entry = cache.get(key)
        if entry and entry.get("start") == start and entry.get("end") == end:
            print("cached", key, len(entry["rows"]), "rows", flush=True)
            continue
        rows, source = fetch_window(
            ARCHIVE_URL, loc["latitude"], loc["longitude"], start, end
        )
        cache[key] = {"start": start, "end": end, "source": source, "rows": rows}
        changed = True
        print("fetched", key, len(rows), "rows", source, flush=True)
        time.sleep(0.6)
    if changed:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=1), encoding="utf8")
    return {k: v["rows"] for k, v in cache.items()}


def compare(cities: list[str], start: str, end: str) -> dict:
    locations = load_locations(cities)
    om_rows = fetch_cached(locations, start, end)
    nasa = pd.read_csv(NASA_WEATHER)
    nasa["date"] = pd.to_datetime(nasa["date"]).dt.strftime("%Y-%m-%d")
    report = {}
    for loc in locations:
        key = loc["location_key"]
        a = nasa[(nasa["location_key"] == key) & (nasa["date"] >= start) & (nasa["date"] <= end)]
        b = pd.DataFrame(om_rows.get(key, []))
        if a.empty or b.empty:
            report[key] = {"error": "no rows", "nasa_rows": int(len(a)), "openmeteo_rows": int(len(b))}
            continue
        m = a.merge(b, on="date", suffixes=("_nasa", "_om"))
        feats = {}
        for f in FEATURES:
            if f + "_nasa" not in m.columns or f + "_om" not in m.columns:
                continue
            x = pd.to_numeric(m[f + "_nasa"], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(m[f + "_om"], errors="coerce").to_numpy(dtype=float)
            ok = ~(np.isnan(x) | np.isnan(y))
            x, y = x[ok], y[ok]
            if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
                continue
            feats[f] = {
                "n": int(len(x)),
                "pearson_r": round(float(np.corrcoef(x, y)[0, 1]), 4),
                "bias_om_minus_nasa": round(float((y - x).mean()), 3),
                "mae": round(float(np.abs(y - x).mean()), 3),
                "mean_ratio_om_over_nasa": round(float((y / x).mean()), 4) if np.all(x != 0) else None,
            }
        report[key] = {"n_days": int(len(m)), "features": feats}
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default="Delhi,Mumbai,Bengaluru,Chennai,Kolkata")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2025-04-30")
    args = ap.parse_args()

    cities = [s.strip() for s in args.cities.split(",") if s.strip()]
    report = {
        "generated_at": date.today().isoformat(),
        "window": [args.start, args.end],
        "note": "Open-Meteo ERA5 archive vs NASA POWER reanalysis, both for the pilot coordinates; wind converted 10m->2m by (2/10)^(1/7)",
        "cities": compare(cities, args.start, args.end),
    }
    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")

    print()
    print("%-22s %-28s %6s %8s %9s %8s" % ("location", "feature", "n", "r", "bias", "mae"))
    for key, block in report["cities"].items():
        for f, v in block.get("features", {}).items():
            print("%-22s %-28s %6d %8.4f %9.3f %8.3f" % (
                key.split("||")[-1][:22], f[:28], v["n"], v["pearson_r"], v["bias_om_minus_nasa"], v["mae"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
