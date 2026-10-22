"""Fetch official CPCB daily city AQI for a date range into a live parallel dataset.

Pipeline (pilot: Delhi, Mumbai, Bengaluru, Chennai, Kolkata):
- Resolves city ids from the CPCB "get-city-wise-avg-aqi-pm-data" master list.
- For each city and each day, calls the daily summary function
  `get-fn-city-aqi-summary-by-time` at two anchors:
    * 15:30 IST (official 4 PM publication value)
    * 17:30 IST (23:00 IST, end-of-day value)
  Both verified against the historical Kaggle dataset on overlap days
  (MAE ~2.4, median 1.0-1.5 for Delhi; see docs/cpcb_recon.md).
- Caches raw responses per city under data/raw/cpcb_live/city_daily/ so reruns
  only fetch missing days.
- Aggregates to a Kaggle-schema CSV (data/raw/cpcb_live/cpcb_daily_aqi.csv)
  WITHOUT touching the research file data/raw/aqi_source/aqi.csv.

Usage:
  python src/fetch_cpcb_live.py --cities "Delhi,Mumbai,Bengaluru,Chennai,Kolkata" \
      --start 2025-05-01 --end 2026-10-21 [--anchors "15:30,17:30"] [--delay 0.1]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cpcb_client import CpcbClient, CpcbTokenSource, QUERY, CpcbError  # noqa: E402

LIVE_DIR = ROOT / "data" / "raw" / "cpcb_live"
CITY_MASTER = LIVE_DIR / "city_master.json"
CACHE_DIR = LIVE_DIR / "city_daily"
DAILY_CSV = LIVE_DIR / "cpcb_daily_aqi.csv"
TOKEN_FILE = LIVE_DIR / "session_token.json"
MANIFEST_FILE = LIVE_DIR / "manifest.json"

SUMMARY_FN = "get-fn-city-aqi-summary-by-time"
LAST3_FN = "get-city-wise-last-3-days-aqi"

AQI_BANDS = [
    (0, 50, "Good"),
    (51, 100, "Satisfactory"),
    (101, 200, "Moderate"),
    (201, 300, "Poor"),
    (301, 400, "Very Poor"),
    (401, 500, "Severe"),
]


def aqi_status(value: float) -> str:
    for lo, hi, name in AQI_BANDS:
        if lo <= value <= hi:
            return name
    return "" if pd.isna(value) else "Very Poor"


def _fmt_date(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def _kaggle_date(d: str) -> str:
    dd = datetime.strptime(d, "%Y-%m-%d")
    return dd.strftime("%d-%m-%Y")


def load_token() -> str:
    """Reuse a fresh token when possible (24h expiry; minted < 23h ago)."""
    tok = ""
    if TOKEN_FILE.exists():
        try:
            info = json.loads(TOKEN_FILE.read_text(encoding="utf8"))
            tok, minted = info.get("token", ""), info.get("minted_at", "")
            if tok and minted:
                age_h = (datetime.now(timezone.utc).replace(tzinfo=None) - datetime.fromisoformat(minted)).total_seconds() / 3600
                if age_h < 23:
                    return tok
        except Exception:  # noqa: BLE001
            pass
        if tok:  # corrupt minted_at but token present; try it, bootstrap will rescue on 401
            return tok
    tok = CpcbTokenSource.from_capture(str(Path(r"C:\Users\cdipi\AppData\Local\Temp\opencode\cpcb-cdp-capture.json")))
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(json.dumps({"token": tok, "saved_at": datetime.now(timezone.utc).isoformat()}), encoding="utf8")
    return tok


def ensure_city_master(client: CpcbClient) -> list[dict]:
    if CITY_MASTER.exists():
        try:
            raw = json.loads(CITY_MASTER.read_text(encoding="utf8"))
            rows = [
                row
                for row in (
                    r if isinstance(r, dict) else dict(zip(["city_id", "city_name", "state_name"], r))
                    for r in raw
                )
                if isinstance(row, dict) and row.get("city_id") and row.get("city_name")
            ]
            if rows:
                return rows
        except Exception:
            pass
    master = client.city_avg_snapshot()
    if not isinstance(master, list) or not master:
        raise CpcbError("could not load city master")
    CITY_MASTER.parent.mkdir(parents=True, exist_ok=True)
    CITY_MASTER.write_text(json.dumps(master, ensure_ascii=False, indent=1), encoding="utf8")
    return master


def resolve_cities(master: list[dict], wanted: list[str]) -> dict[str, str]:
    import re as _re

    def norm(x: str) -> str:
        return _re.sub(r"\s+", " ", x.strip().lower())

    out: dict[str, str] = {}
    missing: list[str] = []
    for want in wanted:
        w = norm(want)
        hits = [r for r in master if norm(r.get("city_name", "")) == w]
        if not hits:
            hits = [r for r in master if w in norm(r.get("city_name", ""))]
        if not hits:  # space-insensitive fallback, e.g. 'Gandhi Nagar' vs 'Gandhinagar'
            hits = [r for r in master if w.replace(" ", "") == norm(r.get("city_name", "")).replace(" ", "")]
        if not hits:
            missing.append(want)
            continue
        out[want.strip()] = hits[0]["city_id"]
    if missing:
        print("cities not in CPCB master (skipped):", missing, flush=True)
    if not out:
        raise CpcbError("none of the requested cities were found in the CPCB master")
    return out


def fetch_daily(client: CpcbClient, city_id: str, day: str, anchors: list[str]) -> dict | None:
    """One day, both anchor calls. Raw payload cached for audit.

    Returns None when no anchor succeeded (day will be retried on rerun).
    """
    out: dict[str, object] = {"date": day}
    ok = False
    for anchor in anchors:
        key = "v_" + anchor.replace(":", "")
        env = cpcb_call_summary(client, city_id, f"{day} {anchor}")
        out[f"pollutants_{anchor.replace(':', '')}"] = ""
        if env is None:
            out[key] = None
            continue
        ok = True
        out[key] = env.get("avg_city_aqi")
        out[f"pollutants_{anchor.replace(':', '')}"] = env.get("top3_pollutants")
    return out if ok else None


def cpcb_call_summary(client: CpcbClient, city_id: str, reftime: str) -> dict | None:
    import requests as _requests

    for attempt in range(5):
        try:
            q = client.post(
                QUERY + SUMMARY_FN + "/execute-procedure",
                {"complexData": {SUMMARY_FN: [{"p_reftime": reftime, "p_city_code": city_id}]}},
            )
            rows = q.get("data") if isinstance(q, dict) else None
            if rows and isinstance(rows, list):
                return rows[0]
            return None
        except _requests.exceptions.ConnectionError:
            # server kills long-lived keep-alive connections; drop the pool and retry
            try:
                client._session.close()
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2.0 + 1.5 * attempt)
        except CpcbError as e:
            msg = str(e)
            if "401" in msg or "403" in msg:
                raise
            time.sleep(1.5 + attempt)
    return None


def city_cache_path(city_id: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{city_id}.json"


def load_and_update_cache(client: CpcbClient, city_id: str, start: date, end: date, anchors: list[str], delay: float) -> list[dict]:
    cache_path = city_cache_path(city_id)
    cache: dict[str, dict] = {}
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf8"))
            raw_data = raw.get("data", [])
            cache = {r["date"]: r for r in raw_data}
        except Exception:
            cache = {}
    fresh = 0
    d = start
    while d <= end:
        day = _fmt_date(d)
        if day not in cache:
            row = fetch_daily(client, city_id, day, anchors)
            if row is not None:
                cache[day] = row
            fresh += 1
            time.sleep(delay)
        d += timedelta(days=1)
    cache_path.write_text(json.dumps({"city_id": city_id, "fetched_at": datetime.now(timezone.utc).isoformat(), "data": list(cache.values())}, indent=1), encoding="utf8")
    return sorted(cache.values(), key=lambda r: r["date"]), fresh


def fetch_last3(client: CpcbClient, city_id: str) -> list[dict]:
    try:
        q = client.post(
            QUERY + LAST3_FN + "/execute-procedure",
            {"complexData": {LAST3_FN: [{"p_city_code": city_id, "p_lat": "", "p_long": ""}]}},
        )
        rows = q.get("data") if isinstance(q, dict) else None
        return rows if isinstance(rows, list) else []
    except Exception:
        return []


def build_csv(city_rows: dict[str, list[dict]], anchors: list[str]) -> pd.DataFrame:
    # Choose 15:30 anchor (official 4PM) as aqi_value; keep 17:30 as secondary.
    primary = anchors[0].replace(":", "")
    secondary = anchors[1].replace(":", "") if len(anchors) > 1 else ""
    frames = []
    for city_id, rows in city_rows.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["city_id"] = city_id
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["aqi_value"] = pd.to_numeric(df.get("v_" + primary), errors="coerce")
    if secondary:
        df["aqi_value_alt_anchor"] = pd.to_numeric(df.get("v_" + secondary), errors="coerce")
        df.loc[df["aqi_value"].isna(), "aqi_value"] = df["aqi_value_alt_anchor"]
    df["air_quality_status"] = df["aqi_value"].map(lambda v: aqi_status(float(v)) if pd.notna(v) else "")
    df["prominent_pollutants"] = df.get(f"pollutants_{primary}", "")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default="Delhi,Mumbai,Bengaluru,Chennai,Kolkata")
    ap.add_argument("--start", default="2025-05-01")
    ap.add_argument("--end", default=_fmt_date(date.today() - timedelta(days=1)))
    ap.add_argument("--anchors", default="15:30,17:30")
    ap.add_argument("--delay", type=float, default=0.08)
    ap.add_argument("--last3-only", action="store_true", help="only refresh the last-3-days probes")
    args = ap.parse_args()

    wanted = [s.strip() for s in args.cities.split(",") if s.strip()]
    anchors = [a.strip() for a in args.anchors.split(",") if a.strip()]
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    token = load_token()
    client = CpcbClient(token, token_store=str(TOKEN_FILE))
    master = ensure_city_master(client)
    city_ids = resolve_cities(master, wanted)
    print("cities:", city_ids, flush=True)

    city_rows: dict[str, list[dict]] = {}
    cache_info: dict[str, dict] = {}
    failures: dict[str, str] = {}
    if not args.last3_only:
        for name, cid in city_ids.items():
            try:
                rows, fresh = load_and_update_cache(client, cid, start, end, anchors, args.delay)
                city_rows[cid] = rows
                cache_info[cid] = {"fresh_calls": fresh, "days": len(rows)}
                print(f"{name}: cached days={len(rows)} fetched_new={fresh}", flush=True)
            except Exception as e:  # noqa: BLE001
                failures[name] = f"{type(e).__name__}: {str(e)[:160]}"
                print(f"{name}: FAILED - {failures[name]} (cache keeps prior days; rerun to resume)", flush=True)

    df = build_csv(city_rows, anchors)
    if df.empty and not args.last3_only:
        if failures:
            print("all cities failed; exiting", flush=True)
        raise SystemExit("no rows collected")
    if not df.empty:
        out = df.copy()
        # map city_id -> city/state names from master
        mmap = {r["city_id"]: r for r in master}
        out["area"] = out["city_id"].map(lambda x: mmap.get(x, {}).get("city_name", ""))
        out["state"] = out["city_id"].map(lambda x: mmap.get(x, {}).get("state_name", ""))
        out["date_kaggle"] = out["date"].map(_kaggle_date)
        out["number_of_monitoring_stations"] = pd.NA
        out["unit"] = "NA"
        out["note"] = ""
        out["source"] = "CPCB_CCR_live"
        out.to_csv(DAILY_CSV, index=False)
        print("wrote", DAILY_CSV, len(out), "rows", flush=True)

    # refresh last-3-days probe (station counts & official series)
    last3: dict[str, dict] = {}
    for name, cid in city_ids.items():
        rows = fetch_last3(client, cid)
        last3[name] = {"city_id": cid, "rows": rows}
        print("last3", name, len(rows), json.dumps(rows[:1], ensure_ascii=False)[:200] if rows else "", flush=True)
    (LIVE_DIR / "last_3_days_aqi.json").write_text(json.dumps(last3, indent=1), encoding="utf8")

    manifest = {
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "aqi": "CPCB CCR portal (airquality.cpcb.gov.in) via service.php proxy",
            "fn": SUMMARY_FN,
            "anchors_ist": anchors,
            "note": "15:30 IST anchor = official 4PM publication; validated vs Kaggle overlap (Delhi MAE 2.4 median 1.5)",
            "station_counts": "only available for the most recent 3 days (last-3-days fn)",
        },
        "cities": city_ids,
        "failures": failures,
        "start": args.start,
        "end": args.end,
        "cache": cache_info,
        "files": {"daily_csv": str(DAILY_CSV.name)},
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=1), encoding="utf8")
    print("manifest written", flush=True)


if __name__ == "__main__":
    main()
