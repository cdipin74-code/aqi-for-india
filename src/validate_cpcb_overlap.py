"""Validate the live CPCB pipeline against the research dataset before any claim.

Two independent checks:

1. AQI-check: fetch `get-fn-city-aqi-summary-by-time` for a HISTORY window that
   overlaps the research dataset (data/raw/aqi_source/aqi.csv, ending
   2025-04-30) and compare city-day values for both anchors. The overlap
   window is fetched fresh and saved under data/raw/cpcb_live/overlap_cache/
   so it never pollutes the live dataset.

2. Weather-check: compare the NASA POWER research weather cache
   (data/interim/weather_daily.csv) with the Open-Meteo live cache
   (data/interim/weather_daily_live.csv) on the overlap dates for the pilot
   location keys; report correlation + bias per feature.

Writes data/raw/cpcb_live/overlap_validation.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cpcb_client import CpcbClient, CpcbTokenSource, QUERY  # noqa: E402

LIVE_DIR = ROOT / "data" / "raw" / "cpcb_live"
OVERLAP_CACHE_DIR = LIVE_DIR / "overlap_cache"
RAW_AQI = ROOT / "data" / "raw" / "aqi_source" / "aqi.csv"
NASA_WEATHER = ROOT / "data" / "interim" / "weather_daily.csv"
OPENMETEO_WEATHER = ROOT / "data" / "interim" / "weather_daily_live.csv"
OUT_JSON = LIVE_DIR / "overlap_validation.json"

SUMMARY_FN = "get-fn-city-aqi-summary-by-time"


def fetch_aqi_overlap(cities: dict[str, str], start: date, end: date, anchors: list[str]) -> dict[str, list[dict]]:
    token = json.loads((LIVE_DIR / "session_token.json").read_text(encoding="utf8"))["token"]
    client = CpcbClient(token, token_store=str(LIVE_DIR / "session_token.json"))
    out: dict[str, list[dict]] = {}
    for name, cid in cities.items():
        cache_path = OVERLAP_CACHE_DIR / f"{cid}.json"
        cache = {}
        if cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf8")).get("data", [])
            cache = {r["date"]: r for r in data}
        d = start
        step = 0
        while d <= end:
            day = d.isoformat()
            step += 1
            if day not in cache:
                row = {"date": day}
                for anchor in anchors:
                    key = "v_" + anchor.replace(":", "")
                    ok = None
                    for attempt in range(4):
                        try:
                            q = client.post(
                                QUERY + SUMMARY_FN + "/execute-procedure",
                                {"complexData": {SUMMARY_FN: [{"p_reftime": f"{day} {anchor}", "p_city_code": cid}]}},
                            )
                            rows = q.get("data") or []
                            ok = rows[0]["avg_city_aqi"] if rows else None
                            row["pollutants_" + anchor.replace(":", "")] = rows[0].get("top3_pollutants") if rows else ""
                            break
                        except Exception:
                            time.sleep(1.0 + attempt)
                            try:
                                client._session.close()
                            except Exception:  # noqa: BLE001
                                pass
                    row[key] = ok
                    time.sleep(0.05)
                # only cache the day when at least one anchor answered
                vals = [r.get("v_" + a.replace(":", "")) for r in [row] for a in anchors]
                if vals and all(v is None for v in vals):
                    print(f"  {name} {day}: both anchors failed; leaving uncached", flush=True)
                else:
                    cache[day] = row
            if step % 30 == 0:
                print(f"  {name}: {step}/{(end - start).days + 1} days", flush=True)
            d += timedelta(days=1)
        OVERLAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"city_id": cid, "data": list(cache.values())}, indent=1), encoding="utf8")
        out[name] = sorted(cache.values(), key=lambda r: r["date"])
        print("aqi overlap cached", name, len(out[name]), flush=True)
    return out


def compare_aqi(overlap: dict[str, list[dict]], anchors: list[str]) -> dict:
    kaggle = pd.read_csv(RAW_AQI)
    kaggle["area_norm"] = kaggle["area"].astype(str).str.strip().str.lower()
    kaggle["date_norm"] = pd.to_datetime(kaggle["date"].astype(str).str.strip(), dayfirst=True, errors="coerce")
    report = {}
    for city, rows in overlap.items():
        kx = kaggle[kaggle["area_norm"] == city.lower()]
        kmap = {r.date_norm.strftime("%Y-%m-%d"): float(r.aqi_value) for r in kx.itertuples() if pd.notna(r.date_norm) and pd.notna(r.aqi_value)}
        out_city = {"n_overlap": 0}
        for anchor in anchors:
            key = "v_" + anchor.replace(":", "")
            errs, live_v, kag_v = [], [], []
            for r in rows:
                day = r["date"]
                if day not in kmap or r.get(key) is None:
                    continue
                lv = float(r[key])
                kv = kmap[day]
                live_v.append(lv)
                kag_v.append(kv)
                errs.append(abs(lv - kv))
            if not errs:
                continue
            e = np.array(errs)
            lv = np.array(live_v)
            kv = np.array(kag_v)
            out_city[anchor] = {
                "n": int(len(e)),
                "mae": round(float(e.mean()), 2),
                "median_abs_err": round(float(np.median(e)), 2),
                "p90_abs_err": round(float(np.percentile(e, 90)), 1),
                "bias_live_minus_kaggle": round(float((lv - kv).mean()), 2),
                "pearson_r": round(float(np.corrcoef(lv, kv)[0, 1]), 4),
                "exact_or_within5": float((e <= 5).mean()) * 100,
            }
            out_city["n_overlap"] = max(out_city["n_overlap"], int(len(e)))
        report[city] = out_city
    return report


WEATHER_FEATURES = [
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


def compare_weather(start: str, end: str) -> dict:
    if not (NASA_WEATHER.exists() and OPENMETEO_WEATHER.exists()):
        return {"error": "weather caches missing"}
    nasa = pd.read_csv(NASA_WEATHER, parse_dates=["date"])
    om = pd.read_csv(OPENMETEO_WEATHER, parse_dates=["date"])
    nasa["date"] = nasa["date"].dt.strftime("%Y-%m-%d")
    om["date"] = om["date"].dt.strftime("%Y-%m-%d")
    keys = sorted(set(om["location_key"].unique()) & set(nasa["location_key"].unique()))
    mask = om["location_key"].isin(keys)
    report = {}
    for key in keys:
        a = nasa[(nasa["location_key"] == key) & (nasa["date"] >= start) & (nasa["date"] <= end)]
        b = om[(om["location_key"] == key) & (om["date"] >= start) & (om["date"] <= end)]
        m = a.merge(b, on=["location_key", "date"], suffixes=("_nasa", "_om"))
        feats = {}
        for f in WEATHER_FEATURES:
            if f + "_nasa" not in m.columns:
                continue
            x = m[f + "_nasa"].astype(float).to_numpy()
            y = m[f + "_om"].astype(float).to_numpy()
            ok = ~(np.isnan(x) | np.isnan(y))
            x, y = x[ok], y[ok]
            if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
                continue
            feats[f] = {
                "n": int(len(x)),
                "pearson_r": round(float(np.corrcoef(x, y)[0, 1]), 4),
                "bias_om_minus_nasa": round(float((y - x).mean()), 3),
                "mae": round(float(np.abs(y - x).mean()), 3),
            }
        report[key] = feats
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default="Delhi,Mumbai,Bengaluru,Chennai,Kolkata")
    ap.add_argument("--start", default="2025-01-01", help="history window for overlap (ends at research-end)")
    ap.add_argument("--end", default="2025-04-30")
    ap.add_argument("--anchors", default="15:30,17:30")
    ap.add_argument("--skip-aqi", action="store_true")
    ap.add_argument("--skip-weather", action="store_true")
    args = ap.parse_args()

    cities_raw = [s.strip() for s in args.cities.split(",") if s.strip()]
    city_master = json.loads((LIVE_DIR / "city_master.json").read_text(encoding="utf8"))
    names = {}
    for r in city_master:
        r = r if isinstance(r, dict) else dict(zip(["city_id", "city_name", "state_name"], r))
        names[r["city_name"].strip().lower()] = r["city_id"]
    cities = {c: names[c.lower()] for c in cities_raw}

    report = {"fetched_at": date.today().isoformat(), "anchors_ist": args.anchors}
    if not args.skip_aqi:
        overlap = fetch_aqi_overlap(cities, date.fromisoformat(args.start), date.fromisoformat(args.end), [a.strip() for a in args.anchors.split(",")])
        report["aqi"] = compare_aqi(overlap, [a.strip() for a in args.anchors.split(",")])
        for city, r in report["aqi"].items():
            line = [f"  {city}: n={r['n_overlap']}"]
            for a in args.anchors.split(","):
                m = r.get(a.strip())
                if m:
                    line.append(f"{a.strip()}IST MAE={m['mae']} med={m['median_abs_err']} r={m['pearson_r']} bias={m['bias_live_minus_kaggle']} within5={m['exact_or_within5']:.0f}%")
            print("\n".join(line))
    if not args.skip_weather:
        report["weather"] = compare_weather(args.start, args.end)
        for key, feats in report["weather"].items():
            print(f"  {key}:")
            for f, v in feats.items():
                print(f"    {f}: r={v['pearson_r']} bias={v['bias_om_minus_nasa']} mae={v['mae']} n={v['n']}")

    OUT_JSON.write_text(json.dumps(report, indent=1), encoding="utf8")
    print("wrote", OUT_JSON)


if __name__ == "__main__":
    main()
