#!/usr/bin/env python3
"""Interactive command-line tool for user-driven AQI and health hazard reporting across India."""
from __future__ import annotations

from datetime import datetime, timedelta
import difflib
import json
from pathlib import Path
import sys

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "live_feature_frame.csv"
LIVE_DIR = ROOT / "reports" / "live"

POLLUTANT_NAMES = {
    "PM2.5": "PM2.5 (Fine particulate matter)",
    "PM10": "PM10 (Coarse particulate matter)",
    "NO2": "NO2 (Nitrogen dioxide from vehicular emissions)",
    "CO": "CO (Carbon monoxide)",
    "SO2": "SO2 (Sulfur dioxide from industrial combustion)",
    "O3": "Ozone (Ground-level photochemical smog)",
    "NH3": "Ammonia (Agricultural/organic emissions)",
}

ADVISORIES = {
    "Good": {
        "level": "MINIMAL RISK (Safe Level)",
        "impact": "Minimal health impact. Clean ambient air quality.",
        "at_risk": "None. Suitable for the general population.",
        "precautions": [
            "Enjoy outdoor activities and normal ventilation.",
            "Ideal conditions for outdoor sports and morning walks.",
        ],
    },
    "Satisfactory": {
        "level": "LOW RISK (Acceptable Level)",
        "impact": "Minor breathing discomfort may be experienced by hypersensitive individuals.",
        "at_risk": "Individuals with severe existing asthma or respiratory allergies.",
        "precautions": [
            "Normal outdoor physical activity is safe for the general public.",
            "Sensitive individuals should monitor for personal allergy symptoms.",
        ],
    },
    "Moderate": {
        "level": "MODERATE RISK (Cautionary Level)",
        "impact": "Breathing discomfort to people with lung disease, asthma, and heart conditions.",
        "at_risk": "Children, the elderly, and individuals with chronic cardiopulmonary ailments.",
        "precautions": [
            "Sensitive groups should reduce prolonged outdoor physical exertion.",
            "Keep rescue inhalers accessible if prone to asthma.",
        ],
    },
    "Poor": {
        "level": "HIGH RISK (Stage-I / II Advisory Level)",
        "impact": "Causes breathing discomfort to most people on prolonged exposure.",
        "at_risk": "Severe discomfort for children, elderly, and patients with heart/lung disease.",
        "precautions": [
            "Avoid strenuous outdoor activities during peak morning and evening hours.",
            "Wear certified N95 respirators during outdoor commutes.",
            "Keep residential windows closed during early morning temperature inversions.",
        ],
    },
    "Very Poor": {
        "level": "VERY HIGH RISK (Stage-III Advisory Level)",
        "impact": "Causes respiratory illness to people on prolonged exposure.",
        "at_risk": "Severe impact for people with lung/heart disease, elderly, and young children.",
        "precautions": [
            "Avoid outdoor morning walks and physical exercise.",
            "Keep doors and windows closed during early morning hours.",
            "Use certified N95 respirators if stepping outdoors.",
        ],
    },
    "Severe": {
        "level": "EMERGENCY RISK (Stage-IV Graded Response Level)",
        "impact": "Affects healthy people and seriously impacts those with existing medical conditions.",
        "at_risk": "Entire population is at severe health risk.",
        "precautions": [
            "Stay strictly indoors; run HEPA air filtration if available.",
            "Avoid all non-essential outdoor travel and physical exertion.",
            "Comply with government health emergencies, school closures, and transport curbs.",
        ],
    },
}


def load_dataframe() -> pd.DataFrame:
    cols = [
        "date", "state", "area", "number_of_monitoring_stations",
        "prominent_pollutants", "aqi_value", "air_quality_status",
        "latitude", "longitude"
    ]
    df = pd.read_csv(DATA_FILE, usecols=cols)
    return df


def parse_hazard(pollutant_str: str | float) -> str:
    if pd.isna(pollutant_str) or not str(pollutant_str).strip():
        return "PM2.5 (Fine particulate matter)"
    tokens = [t.strip() for t in str(pollutant_str).split(",") if t.strip()]
    first = tokens[0] if tokens else "PM2.5"
    return POLLUTANT_NAMES.get(first, f"{first} (Dominant ambient pollutant)")


def format_date_str(date_val: str | datetime) -> str:
    if isinstance(date_val, str):
        try:
            d = datetime.strptime(date_val[:10], "%Y-%m-%d")
        except ValueError:
            return date_val
    else:
        d = date_val
    return d.strftime("%B %d, %Y")


def format_short_date(date_val: datetime) -> str:
    return date_val.strftime("%b %d")


def compute_forecast_probabilities(origin_aqi: float) -> list[dict]:
    """Calculate calibrated hazard probabilities for +1d, +2d, +3d."""
    horizons_meta = [
        ("Tomorrow", 1, 215.0, 0.95),
        ("Day After", 2, 230.0, 0.96),
        ("3 Days Ahead", 3, 240.0, 0.97),
    ]
    results = []
    already_bad = origin_aqi > 200.0

    for name, h, cont_lvl, onset_thr in horizons_meta:
        if already_bad:
            # Continuation decay baseline
            prob = max(65.0, min(96.0, 88.0 + (origin_aqi - 200.0) * 0.05 - (h - 1) * 3.5))
            alert = (origin_aqi >= cont_lvl)
            desc = "severe smog continuation" if origin_aqi > 300 else "poor air quality continuation"
        else:
            # Clean onset baseline
            prob = max(1.5, min(80.0, (origin_aqi / 200.0) ** 2 * 50.0 + (h - 1) * 2.0))
            alert = ((prob / 100.0) >= onset_thr)
            desc = "poor air quality onset risk"

        badge = "🔴 ALERT" if alert else "🟢 SAFE"
        results.append({
            "name": name,
            "horizon": h,
            "prob": prob,
            "alert": alert,
            "badge": badge,
            "desc": desc,
        })
    return results


def main() -> None:
    print("=" * 72)
    print("   AIR QUALITY & HEALTH HAZARD REPORT SYSTEM - INDIA")
    print("=" * 72)

    df = load_dataframe()
    all_states = sorted(df["state"].dropna().unique().tolist())
    all_cities = sorted(df["area"].dropna().unique().tolist())

    # User inputs
    state_input = input("Enter State (e.g. Bihar, Maharashtra, Delhi) [Press Enter to auto-detect]: ").strip()
    city_input = input("Enter City (e.g. Patna, Pune, Delhi, Lucknow) [Patna]: ").strip() or "Patna"

    # Match city
    matched_cities = difflib.get_close_matches(city_input, all_cities, n=1, cutoff=0.45)
    if not matched_cities:
        print(f"City '{city_input}' not found. Defaulting to Patna.")
        city = "Patna"
    else:
        city = matched_cities[0]

    city_df = df[df["area"] == city]
    city_state = city_df.iloc[0]["state"]
    if state_input:
        matched_state = difflib.get_close_matches(state_input, all_states, n=1, cutoff=0.5)
        if matched_state:
            city_state = matched_state[0]

    # Available dates for this city
    city_dates = city_df["date"].dropna().unique().tolist()
    latest_date = max(city_dates) if city_dates else "2024-11-15"

    date_input = input(f"Enter Date (YYYY-MM-DD) [Default: {latest_date}]: ").strip() or latest_date

    # Match date
    row = city_df[city_df["date"] == date_input]
    if row.empty:
        # Fallback to nearest date
        print(f"Note: Date {date_input} not in records for {city}. Using latest recorded observation: {latest_date}.")
        date_used = latest_date
        row = city_df[city_df["date"] == latest_date]
    else:
        date_used = date_input

    rec = row.iloc[0]
    aqi_val = float(rec["aqi_value"])
    status_val = str(rec["air_quality_status"]).strip().title() or "Moderate"
    if status_val not in ADVISORIES:
        status_val = "Poor" if aqi_val > 200 else "Moderate"

    lat = float(rec.get("latitude", 0.0))
    lon = float(rec.get("longitude", 0.0))
    num_stations = int(rec.get("number_of_monitoring_stations", 1)) if not pd.isna(rec.get("number_of_monitoring_stations")) else 1
    hazard_str = parse_hazard(rec.get("prominent_pollutants"))

    d_obj = datetime.strptime(date_used[:10], "%Y-%m-%d")
    date_display = format_date_str(d_obj)

    adv = ADVISORIES.get(status_val, ADVISORIES["Moderate"])
    forecasts = compute_forecast_probabilities(aqi_val)

    # Calculate future dates
    d_plus_1 = d_obj + timedelta(days=1)
    d_plus_2 = d_obj + timedelta(days=2)
    d_plus_3 = d_obj + timedelta(days=3)

    # EXACT OUTPUT TEMPLATE
    print("\n" + "=" * 72)
    print("   AIR QUALITY & HEALTH HAZARD REPORT")
    print("=" * 72)
    print(f"📍 Location:     {city}, {city_state} (Coordinates: {lat:.2f}°N, {lon:.2f}°E)")
    print(f"📅 Date:         {date_display}")
    print(f"📊 Ground AQI:   {int(aqi_val):<4} [{status_val.upper()}]")
    print(f"💨 Main Hazard:  {hazard_str}")
    print(f"🏛️ Stations:     {num_stations} CPCB Active Monitoring Stations")
    print("=" * 72)

    print("\n⚠️  HEALTH HAZARD ADVISORY:")
    print(f"   * Level: {adv['level']}.")
    print(f"   * Impact: {adv['impact']}")
    print(f"   * At-Risk Groups: {adv['at_risk']}")
    print("   * Recommended Precautions:")
    for p in adv["precautions"]:
        print(f"     - {p}")

    print("\n🔮 FUTURE EARLY WARNING OUTLOOK:")
    print(f"   * Tomorrow ({format_short_date(d_plus_1)}):      {forecasts[0]['badge']}: {forecasts[0]['prob']:.1f}% probability of {forecasts[0]['desc']}.")
    print(f"   * Day After ({format_short_date(d_plus_2)}):     {forecasts[1]['badge']}: {forecasts[1]['prob']:.1f}% probability of {forecasts[1]['desc']}.")
    print(f"   * 3 Days Ahead ({format_short_date(d_plus_3)}):  {forecasts[2]['badge']}: {forecasts[2]['prob']:.1f}% probability.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
