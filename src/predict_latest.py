"""Issue next-1/2/3-day AQI early warnings for the pilot cities with live data.

Operates strictly on information available at issue time t (the predictor set is
identical to training; no target-window or future weather values are used):

1. research feature frame + CPCB live rows + live weather
       (data/processed/live_feature_frame.csv, built by build_live_dataset.py)
2. exactly the training feature path:
       train_regional_early_warning.build_feature_table()      [DATA_PATH patched]
       evaluate_onset_warning.add_neighbour_features(k=5)
3. saved models:
       models/regional_early_warning/global_aqi_gt_200_h{h}.joblib   (any origin)
       models/onset/onset_aqi_gt_200_h{h}.joblib                     (clean origins)
4. the paper's jointly calibrated two-head rule (reports/onset/
   combined_system_calibrated.csv, rows 'two-head, jointly calibrated'):
       continuation head: alert iff AQI(t) >= level_h      (already above limit)
       onset head:        alert iff p_onset >= thr_h        (clean origins)
       joint alert = continuation OR onset

Weather caveat is reported per city: research rows carry reanalysis values,
live rows carry Open-Meteo (ERA5 archive, forecast API for recent days),
so weather-based features switch source inside the lag chains.

Usage:
  python src/predict_latest.py [--asof YYYY-MM-DD] [--cities "Delhi,Mumbai,..."]
Outputs:
  reports/live/predictions_<asof>.json + a console table (ASCII only).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train_regional_early_warning as tre  # noqa: E402
import evaluate_onset_warning as eow  # noqa: E402

FRAME_PATH = ROOT / "data" / "processed" / "live_feature_frame.csv"
MODEL_DIR = ROOT / "models"
CALIB_TABLE = ROOT / "reports" / "onset" / "combined_system_calibrated.csv"
OUT_DIR = ROOT / "reports" / "live"

BANDS = [(0, 50, "Good"), (51, 100, "Satisfactory"), (101, 200, "Moderate"),
         (201, 300, "Poor"), (301, 400, "Very Poor"), (401, 1000, "Severe")]


def aqi_band(value: float) -> str:
    for lo, hi, name in BANDS:
        if lo <= value <= hi:
            return name
    return "Out of range"


def calibrated_pairs() -> dict[int, dict[str, float]]:
    """horizon -> {continuation_level, onset_threshold} of the published rule."""
    table = pd.read_csv(CALIB_TABLE)
    rows = table[table["system"].str.startswith("two-head")]
    out: dict[int, dict[str, float]] = {}
    for _, r in rows.iterrows():
        out[int(r["horizon_days"])] = {
            "continuation_level": float(r["continuation_level"]),
            "onset_threshold": float(r["onset_threshold"]),
        }
    return out


def load_models():
    """per horizon: global pipeline, onset pipeline, threshold files."""
    global_models, onset_models, meta = {}, {}, {}
    for h in (1, 2, 3):
        gp = MODEL_DIR / "regional_early_warning" / f"global_aqi_gt_200_h{h}.joblib"
        op = MODEL_DIR / "onset" / f"onset_aqi_gt_200_h{h}.joblib"
        gmeta = json.loads((MODEL_DIR / "regional_early_warning" / f"global_aqi_gt_200_h{h}_threshold.json").read_text(encoding="utf8"))
        ometa = json.loads((MODEL_DIR / "onset" / f"onset_aqi_gt_200_h{h}_threshold.json").read_text(encoding="utf8"))
        gpipeline = joblib.load(gp)
        if isinstance(gpipeline, dict) and "pipeline" in gpipeline:
            gpipeline = gpipeline["pipeline"]
        global_models[h] = gpipeline
        onset_models[h] = joblib.load(op)
        meta[h] = {
            "global_numeric": gmeta["numeric_features"],
            "global_threshold": float(gmeta["alert_threshold"]),
            "onset_numeric": ometa["numeric_features"],
            "onset_threshold_model": float(ometa["alert_threshold"]),
        }
    return global_models, onset_models, meta


def issue_date(frame: pd.DataFrame, pilot_keys: list[str]) -> str:
    sub = frame[frame["location_key"].isin(pilot_keys) & frame["aqi_at_issue"].notna()]
    return str(sub["date"].max().date())


def cast_row(row: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    X = row[numeric + categorical].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in categorical:
        X[c] = X[c].astype(str)
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="issue date (YYYY-MM-DD); default = latest live day")
    ap.add_argument("--cities", default="all", help='comma list or "all"')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1-2) training-identical feature path over the live frame
    tre.DATA_PATH = FRAME_PATH
    frame, numeric_base, categorical = tre.build_feature_table()
    frame, neighbour_features = eow.add_neighbour_features(frame)

    plan = json.loads((ROOT / "data" / "raw" / "cpcb_live" / "neighbour_plan.json").read_text(encoding="utf8"))
    pilot_keys = {name: p["pilot_location_key"] for name, p in plan.items()}
    pilot_cities = {} if not args.cities or args.cities == "all" else [n.strip() for n in args.cities.split(",") if n.strip()]
    wanted = pilot_cities or list(pilot_keys)

    asof = args.asof or issue_date(frame, list(pilot_keys.values()))
    asof_ts = pd.Timestamp(asof)

    global_models, onset_models, meta = load_models()
    calib = calibrated_pairs()

    results: dict[str, dict] = {}
    for city in wanted:
        key = pilot_keys.get(city)
        if key is None:
            results[city] = {"error": f"not a planned pilot; plan covers {list(pilot_keys)}"}
            continue
        row = frame[(frame["location_key"] == key) & (frame["date"] == asof_ts)]
        if row.empty or pd.isna(row.iloc[0]["aqi_at_issue"]):
            results[city] = {"error": f"no live AQI row for issue date {asof}"}
            continue
        city_out: dict[str, object] = {
            "location_key": key,
            "issue_date": asof,
            "aqi_at_issue": float(row.iloc[0]["aqi_at_issue"]),
            "status_at_issue": aqi_band(float(row.iloc[0]["aqi_at_issue"])),
            "weather_source_at_issue": row.iloc[0].get("weather_source", ""),
            "horizons": {},
        }
        for h in (1, 2, 3):
            g_num = meta[h]["global_numeric"]
            o_num = meta[h]["onset_numeric"]
            p_global = float(global_models[h].predict_proba(cast_row(row, g_num, categorical))[:, 1][0])
            p_onset = float(onset_models[h].predict_proba(cast_row(row, o_num, categorical))[:, 1][0])

            level = calib[h]["continuation_level"]
            thr_on_cal = calib[h]["onset_threshold"]
            aqi = float(row.iloc[0]["aqi_at_issue"])
            already_bad = aqi > 200.0
            continuation_alert = aqi >= level
            onset_alert = (not already_bad) and (p_onset >= thr_on_cal)
            joint_alert = bool(continuation_alert or onset_alert)

            city_out["horizons"][f"+{h}d"] = {
                "target_date": str((asof_ts + pd.Timedelta(days=h)).date()),
                "p_hazard_global": round(p_global, 4),
                "p_onset_clean": round(p_onset, 4),
                "relevant_head": "continuation" if already_bad else "onset",
                "joint_alert": joint_alert,
                "rule": {
                    "continuation_level": level,
                    "onset_threshold": thr_on_cal,
                    "alert_threshold_global_model": meta[h]["global_threshold"],
                    "onset_threshold_model": meta[h]["onset_threshold_model"],
                },
                "global_threshold_alert": bool(p_global >= meta[h]["global_threshold"]),
                "naive_persistence_aqi": aqi,
            }
        # neighbour coverage at issue day (+1d for the change feature)
        nd = eow.NEIGHBOURS
        nb_mean_a, nb_mean_b = row.iloc[0].get("neighbour_mean_aqi"), row.iloc[0].get("neighbour_count_reporting")
        city_out["neighbour_context"] = {
            "neighbour_count_reporting": None if pd.isna(nb_mean_b) else float(nb_mean_b),
            "neighbour_mean_aqi": None if pd.isna(nb_mean_a) else float(nb_mean_a),
            "expected": nd,
            "note": "mean/max AQI across the 5 nearest research areas; CPCB city "
                    "level feeds substitute for research areas in the plan file",
        }
        results[city] = city_out

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "asof": asof,
        "system": "jointly calibrated two-head rule (continuation | onset), support: global hazard head",
        "models": {
            "feature_path": "tre.build_feature_table + eow.add_neighbour_features(k=5)",
            "weather_note": "live rows use Open-Meteo (ERA5 archive; recent days forecast API); "
                            "research rows reanalysis; lag chains may switch source",
        },
        "calibrated_rule": calib,
        "pilots": results,
    }
    outfile = OUT_DIR / f"predictions_{asof}.json"
    outfile.write_text(json.dumps(payload, indent=1), encoding="utf8")

    # console
    print("ASOF", asof, "| joint rule h1: cont>=", calib[1]["continuation_level"],
          "onset>=", calib[1]["onset_threshold"], "| h2:", calib[2], "| h3:", calib[3])
    for city, r in results.items():
        if "error" in r:
            print(f"{city:10s} ERROR {r['error']}")
            continue
        aqi = r["aqi_at_issue"]
        line = f"{city:10s} AQI(t)={aqi:5.0f} {r['status_at_issue']:12s} nb={r['neighbour_context']['neighbour_count_reporting']}"
        print(line)
        for h in (1, 2, 3):
            hh = r["horizons"][f"+{h}d"]
            print(f"          +{h}d p_glob={hh['p_hazard_global']:.3f} p_onset={hh['p_onset_clean']:.3f} "
                  f"head={hh['relevant_head']:12s} ALERT={hh['joint_alert']}")
    print("wrote", outfile)


if __name__ == "__main__":
    main()
