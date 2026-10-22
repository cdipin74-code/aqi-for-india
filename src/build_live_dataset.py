"""Build the live feature frame: research rows + CPCB live rows + live weather.

Reads (never written to):
  data/processed/aqi_prediction_full.csv     research feature base (ends 2025-04-30)
  data/raw/cpcb_live/city_daily/<city>.json  CPCB live daily caches (2025-05-01 ..)
  data/interim/weather_daily_live.csv        Open-Meteo weather for the gap
  data/raw/cpcb_live/neighbour_plan.json     pilot <-> neighbour mapping

Writes:
  data/processed/live_feature_frame.csv      research rows + live rows, aligned
                                             to the research column layout
  data/processed/live_feature_frame_report.json

The file is a SUPPLEMENT to the research frame (which stays untouched); it is
exactly what train_regional_early_warning.build_feature_table() expects, with
DATA_PATH monkeypatched to this file by predict_latest.py.

Weather provenance: research rows carry the research pipeline's reanalysis
values; live rows carry Open-Meteo (ERA5 archive for past days, forecast API
for the most recent days).  Source kept in 'weather_source'.

AQI provenance: research rows Kaggle/CPCB historical; live rows CPCB CCR
daily-summary (15:30 IST anchor = official 4 PM publication; fallback 17:30).

Models only consume aqi_at_issue/lat/lon + derived lags + weather lags +
region/state/location_key, so pollutant concentrations and metadata columns
(season, prominence flags, ...) are left missing on live rows deliberately.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RESEARCH = ROOT / "data" / "processed" / "aqi_prediction_full.csv"
LIVE_DIR = ROOT / "data" / "raw" / "cpcb_live"
CACHE_DIR = LIVE_DIR / "city_daily"
WEATHER = ROOT / "data" / "interim" / "weather_daily_live.csv"
PLAN = LIVE_DIR / "neighbour_plan.json"
OUT = ROOT / "data" / "processed" / "live_feature_frame.csv"
OUT_REPORT = ROOT / "data" / "processed" / "live_feature_frame_report.json"

WEATHER_COLS = (
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "relative_humidity_2m_mean", "precipitation_sum", "wind_speed_2m_mean",
    "wind_speed_2m_max", "surface_pressure_mean", "shortwave_radiation_sum",
)


def aqi_status(value: float) -> str:
    if pd.isna(value):
        return ""
    v = float(value)
    for lo, hi, name in [(0, 50, "Good"), (51, 100, "Satisfactory"), (101, 200, "Moderate"),
                         (201, 300, "Poor"), (301, 400, "Very Poor")]:
        if lo <= v <= hi:
            return name
    return "Severe" if v > 400 else "Out of range"


def cache_rows(city_id: str) -> list[dict]:
    path = CACHE_DIR / f"{city_id}.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf8"))
    return [r for r in raw.get("data", []) if isinstance(r, dict) and r.get("date")]


def load_plan() -> dict:
    return json.loads(PLAN.read_text(encoding="utf8"))


def area_infos(plan: dict) -> dict[str, dict]:
    """location_key -> static info; pilots carry the full live range, neighbours
    are trimmed later to the recent window."""
    out: dict[str, dict] = {}
    for pilot_name, p in plan.items():
        key = p["pilot_location_key"]
        out[key] = {
            "area": pilot_name, "state": p["state"], "latitude": p["latitude"],
            "longitude": p["longitude"], "city_id": p["cpcb_city_id"], "role": "pilot",
        }
    return out


def resolve_neighbours(plan: dict, neighbour_window_days: int) -> list[dict]:
    window_start = (date.fromisoformat(latest_live_date()) - timedelta(days=neighbour_window_days)).isoformat()
    infos, seen_keys = [], set()
    for pilot_name, p in plan.items():
        for n in p["neighbours"]:
            if not n["cpcb_city_id"] or n["location_key"] in seen_keys:
                continue
            seen_keys.add(n["location_key"])
            infos.append({
                "area": n["area"], "state": n["state"], "latitude": n["latitude"],
                "longitude": n["longitude"], "city_id": n["cpcb_city_id"], "role": "neighbour",
                "window_start": window_start,
            })
    return infos


def latest_live_date() -> str:
    latest = date.min
    for path in CACHE_DIR.glob("CTY-*.json"):
        try:
            rows = json.loads(path.read_text(encoding="utf8")).get("data", [])
        except Exception:  # noqa: BLE001
            continue
        for r in rows:
            if isinstance(r, dict) and r.get("date") and r["date"] > latest.isoformat():
                latest = date.fromisoformat(r["date"])
    if latest == date.min:
        raise SystemExit("no live cache rows; run fetch_cpcb_live.py first")
    return latest.isoformat()


def live_rows(infos: list[dict], start: str, end: str) -> pd.DataFrame:
    cache: dict[str, list[dict]] = {}
    for info in infos:
        cache[info["city_id"]] = cache_rows(info["city_id"])

    rows: list[dict] = []
    per_area: dict[str, dict] = {}
    for info in infos:
        lo = info["window_start"] if info["role"] == "neighbour" else start
        count = 0
        for r in cache[info["city_id"]]:
            d = r["date"]
            if d < lo or d > end:
                continue
            aqi = pd.to_numeric(r.get("v_1530"), errors="coerce")
            if pd.isna(aqi):
                aqi = pd.to_numeric(r.get("v_1730"), errors="coerce")
            rows.append({
                "date": d,
                "state": info["state"],
                "area": info["area"],
                "number_of_monitoring_stations": pd.NA,
                "prominent_pollutants": r.get("pollutants_1530") or "",
                "aqi_value": aqi,
                "air_quality_status": aqi_status(aqi),
                "latitude": info["latitude"],
                "longitude": info["longitude"],
                "role": info["role"],
                "live_city_id": info["city_id"],
            })
            count += 1
        per_area[info["area"]] = {"role": info["role"], "rows": count,
                                  "from": lo, "to": end, "city_id": info["city_id"]}
    df = pd.DataFrame(rows)
    stats = {"per_area": per_area}
    return df, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-05-01", help="first live date (pilots)")
    ap.add_argument("--neighbour-window-days", type=int, default=40,
                    help="days back from latest cached date for neighbour rows")
    args = ap.parse_args()

    research = pd.read_csv(RESEARCH, parse_dates=["date"])
    research.columns = [str(c).strip() for c in research.columns]
    research["date"] = research["date"].dt.strftime("%Y-%m-%d")  # uniform ISO day strings

    end = latest_live_date()
    plan = load_plan()
    infos = list(area_infos(plan).values())
    infos.extend(resolve_neighbours(plan, args.neighbour_window_days))
    live, stats = live_rows(infos, args.start, end)
    live["location_key"] = live["state"].str.strip() + "||" + live["area"].str.strip()

    # live weather join (on the same location_key the SPA of the research uses)
    w = pd.read_csv(WEATHER, parse_dates=["date"])
    w["date_s"] = w["date"].dt.strftime("%Y-%m-%d")
    w = w[["location_key", "date_s", *WEATHER_COLS, "weather_source"]]
    live = live.merge(w, how="left", left_on=["location_key", "date"], right_on=["location_key", "date_s"])
    live = live.drop(columns=["date_s"])

    # align to research column layout (+ extra bookkeeping columns)
    cols = list(research.columns)
    extra = ["role", "live_city_id", "weather_source"]
    for c in cols:
        if c not in live.columns:
            live[c] = pd.NA
    for c in extra:
        if c not in live.columns:
            live[c] = pd.NA
    live_out = live[cols + extra].copy()

    research_out = research.copy()
    research_out["role"] = "research"
    for c in ["live_city_id", "weather_source"]:
        research_out[c] = pd.NA
    for c in extra:
        if c not in research_out.columns:
            research_out[c] = "research" if c == "role" else pd.NA
    research_out["weather_source"] = "research_reanalysis"

    full = pd.concat([research_out[cols + extra], live_out[cols + extra]], ignore_index=True)
    full = full.drop_duplicates(subset=["state", "area", "date"], keep="first").reset_index(drop=True)
    full.to_csv(OUT, index=False)

    cov = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(full)),
        "research_rows": int(len(research_out)),
        "live_rows": int(len(live_out)),
        "live_date_range": [str(live["date"].min()), str(live["date"].max())] if not live.empty else None,
        "areas_in_frame": int(full.groupby(["state", "area"]).ngroups),
        "neighbour_window_days": args.neighbour_window_days,
        "coverage_per_area": stats["per_area"],
        "columns_added_by_stream": extra,
    }
    OUT_REPORT.write_text(json.dumps(cov, indent=1), encoding="utf8")
    print("wrote", OUT, len(full), "rows (research", len(research_out), "+ live", len(live_out), ")")
    for area, v in stats["per_area"].items():
        print("  %-16s %-9s rows=%-4d %s..%s (%s)" % (area, v["role"], v["rows"], v["from"], v["to"], v["city_id"]))


if __name__ == "__main__":
    main()
