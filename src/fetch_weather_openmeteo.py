"""Fetch daily weather for the live dataset from Open-Meteo (keyless).

Two windows, merged:
- historical archive (ERA5 reanalysis): archive-api.open-meteo.com/v1/archive
- recent + next days (forecast model): api.open-meteo.com/v1/forecast with
  past_days / forecast_days (supports dates newer than the archive's lag)

Fields are mapped to the NASA POWER names used by build_dataset.py so the
feature pipeline is source-agnostic. Two unit conversions are mandatory:

- Wind: Open-Meteo defaults to km/h. We request ``wind_speed_unit=ms`` and
  then convert 10 m -> 2 m with the power-law profile factor
  (2/10)^(1/7) ~= 0.759, because NASA POWER WS2M is a 2 m wind in m/s.
- Shortwave radiation: Open-Meteo returns MJ/m^2/day, NASA POWER
  ALLSKY_SFC_SW_DWN is kWh/m^2/day, so the value is divided by 3.6.

Both conversions are checked against NASA POWER on overlap days by
src/validate_weather_overlap.py (see docs/cpcb_recon.md).

Outputs (research caches are NOT modified):
- data/interim/weather_daily_live.csv  (only these rows + manifest)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
GEOCODE_CACHE = INTERIM / "geocoded_locations.csv"
OUT_CSV = INTERIM / "weather_daily_live.csv"
OUT_MANIFEST = LIVE_DIR if (LIVE_DIR := ROOT / "data" / "raw" / "cpcb_live") else None

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DAILY_VARS = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_mean",
    "wind_speed_10m_max",
    "surface_pressure_mean",
    "shortwave_radiation_sum",
]

W2M_FACTOR = (2.0 / 10.0) ** (1.0 / 7.0)  # documented power-law conversion
MJ_PER_KWH = 3.6  # Open-Meteo shortwave (MJ/m2) -> NASA POWER (kWh/m2)


def resolve_locations(cities: list[str]) -> list[dict[str, Any]]:
    """Map city names to (state, city, lat, lon, location_key) using:
    1) research geocode cache (keeps the same location_key as training data)
    2) CPCB master list coordinates (saved in data/raw/cpcb_live/city_master.json)
    """
    out = []
    done = set()
    if GEOCODE_CACHE.exists():
        geo = pd.read_csv(GEOCODE_CACHE)
        for want in cities:
            m = geo[geo["area"].str.strip().str.lower() == want.strip().lower()]
            if not m.empty:
                r = m.iloc[0]
                out.append({
                    "city": want,
                    "state": r["state"],
                    "location_key": r["location_key"],
                    "latitude": float(r["latitude"]),
                    "longitude": float(r["longitude"]),
                })
                done.add(want.strip().lower())
    master_path = ROOT / "data" / "raw" / "cpcb_live" / "city_master.json"
    if master_path.exists():
        master = json.loads(master_path.read_text(encoding="utf8"))
        names = {}
        for r in master:
            r = r if isinstance(r, dict) else dict(zip(["city_id", "city_name", "state_name"], r))
            names[r["city_name"].strip().lower()] = r
        for want in cities:
            if want.strip().lower() in done:
                continue
            r = names.get(want.strip().lower())
            if r is None:
                print("WARN: no location for", want)
                continue
            states = {
                "Maharashtra": "Maharashtra",
                "Karnataka": "Karnataka",
                "Tamil Nadu": "Tamil Nadu",
                "West Bengal": "West Bengal",
                "Delhi": "Delhi",
            }
            state = r.get("state_name", "")
            out.append({
                "city": want.strip(),
                "state": state,
                "location_key": f"{state}||{want.strip()}" if not (state == "Delhi" and want.strip().lower() == "delhi") else "Delhi||Delhi",
                "latitude": float(r["latitude"]),
                "longitude": float(r["longitude"]),
            })
    return out


def fetch_window(url: str, lat: float, lon: float, start: str, end: str, past_days: int = 0, forecast_days: int = 0) -> tuple[list[dict], str]:
    params2: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Kolkata",
        "wind_speed_unit": "ms",
        "start_date": start,
        "end_date": end,
    }
    if url.endswith("/forecast"):
        params2.pop("start_date")
        params2.pop("end_date")
        params2["past_days"] = past_days
        params2["forecast_days"] = forecast_days
    last_err: Exception | None = None
    r = None
    for attempt in range(4):
        try:
            r = requests.get(url, params=params2, timeout=90)
            if r.status_code == 200:
                break
            last_err = RuntimeError(f"open-meteo {r.status_code}: {r.text[:160]}")
        except Exception as err:  # noqa: BLE001
            last_err = err
        time.sleep(1.5 * (attempt + 1))
    if r is None or r.status_code != 200:
        raise RuntimeError(str(last_err))
    data = r.json()
    daily = data.get("daily") or {}
    if not daily:
        return [], "OPEN_METEO_" + ("FORECAST" if url.endswith("/forecast") else "ERA5")
    n = len(daily["time"])
    rows = []
    src = "OPEN_METEO_FORECAST" if url.endswith("/forecast") else "OPEN_METEO_ERA5"
    for i, d in enumerate(daily["time"]):
        ws_mean = daily.get("wind_speed_10m_mean", [None] * n)[i]
        ws_max = daily.get("wind_speed_10m_max", [None] * n)[i]
        rows.append({
            "date": d,
            "temperature_2m_mean": daily.get("temperature_2m_mean", [None] * n)[i],
            "temperature_2m_max": daily.get("temperature_2m_max", [None] * n)[i],
            "temperature_2m_min": daily.get("temperature_2m_min", [None] * n)[i],
            "relative_humidity_2m_mean": daily.get("relative_humidity_2m_mean", [None] * n)[i],
            "precipitation_sum": daily.get("precipitation_sum", [None] * n)[i],
            "wind_speed_2m_mean": None if ws_mean is None else round(ws_mean * W2M_FACTOR, 4),
            "wind_speed_2m_max": None if ws_max is None else round(ws_max * W2M_FACTOR, 4),
            "surface_pressure_mean": daily.get("surface_pressure_mean", [None] * n)[i],
            "shortwave_radiation_sum": None if daily.get("shortwave_radiation_sum", [None] * n)[i] is None
            else round(daily["shortwave_radiation_sum"][i] / MJ_PER_KWH, 4),
        })
    return rows, src


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default="Delhi,Mumbai,Bengaluru,Chennai,Kolkata")
    ap.add_argument("--start", default="2025-05-01")
    ap.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    ap.add_argument("--forecast-days", type=int, default=4, help="keep up to N forecast days beyond today")
    args = ap.parse_args()

    wanted = [s.strip() for s in args.cities.split(",") if s.strip()]
    locs = resolve_locations(wanted)
    print("locations:", [(l["city"], l["location_key"]) for l in locs], flush=True)
    if not locs:
        raise SystemExit("no locations resolved")

    all_frames = []
    manifest = {
        "fetched_at_utc": datetime.utcnow().isoformat(),
        "wind_2m_conversion": "Open-Meteo wind requested in m/s, then 10 m * (2/10)^(1/7) = 0.759 to match NASA POWER WS2M",
        "shortwave_conversion": "Open-Meteo MJ/m2/day divided by 3.6 to match NASA POWER ALLSKY_SFC_SW_DWN (kWh/m2/day)",
        "archive_lag_days": "ERA5 archive typically lags ~5 days; recent days come from the forecast API past_days",
        "locations": [],
    }
    for loc in locs:
        rows_archive: list[dict] = []
        rows_recent: list[dict] = []
        # archive: start..(end - 6 days)
        archive_end = date.fromisoformat(args.end) - timedelta(days=6)
        if archive_end >= date.fromisoformat(args.start):
            rows_archive, _ = fetch_window(ARCHIVE_URL, loc["latitude"], loc["longitude"], args.start, archive_end.isoformat())
        # forecast API: last 10 days .. +forecast_days (covers archive lag window)
        f_start_date = date.today() - timedelta(days=10)
        rows_recent, _ = fetch_window(
            FORECAST_URL,
            loc["latitude"],
            loc["longitude"],
            f_start_date.isoformat(),
            (date.today() + timedelta(days=args.forecast_days)).isoformat(),
            past_days=10,
            forecast_days=args.forecast_days,
        )
        seen = set()
        rows = []
        for base in rows_archive + rows_recent:
            d = base["date"]
            if d in seen:
                continue
            seen.add(d)
            rows.append(base)
        df = pd.DataFrame(rows)
        if df.empty:
            print("WARN: no weather for", loc["city"])
            continue
        df["location_key"] = loc["location_key"]
        df["state"] = loc["state"]
        df["city"] = loc["city"]
        all_frames.append(df)
        manifest["locations"].append({
            **loc,
            "rows": int(len(df)),
            "date_min": str(df["date"].min()) if not df.empty else None,
            "date_max": str(df["date"].max()) if not df.empty else None,
        })
        print(loc["city"], "rows", len(df), df["date"].min(), "->", df["date"].max(), flush=True)
        time.sleep(0.6)

    if not all_frames:
        raise SystemExit("no weather rows")
    big = pd.concat(all_frames, ignore_index=True)
    big["weather_source"] = "OPEN_METEO"
    cols = [
        "location_key",
        "date",
        "city",
        "state",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "wind_speed_2m_mean",
        "wind_speed_2m_max",
        "surface_pressure_mean",
        "shortwave_radiation_sum",
        "weather_source",
    ]
    big = big[cols].sort_values(["location_key", "date"])
    INTERIM.mkdir(parents=True, exist_ok=True)
    big.to_csv(OUT_CSV, index=False)
    print("wrote", OUT_CSV, len(big), "rows", flush=True)
    (ROOT / "data" / "raw" / "cpcb_live" / "weather_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
