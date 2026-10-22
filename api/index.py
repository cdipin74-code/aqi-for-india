from http.server import BaseHTTPRequestHandler
import json
from pathlib import Path
import sqlite3
from urllib.parse import parse_qs, urlparse
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(__file__).resolve().parent / "aqi_light.db"
if not DB_PATH.exists():
    DB_PATH = ROOT / "data" / "aqi_light.db"
CITIES_PATH = Path(__file__).resolve().parent.parent / "public" / "cities.json"

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


def parse_hazard(pollutant_str: str | None) -> str:
    if not pollutant_str or not str(pollutant_str).strip():
        return "PM2.5 (Fine particulate matter)"
    tokens = [t.strip() for t in str(pollutant_str).split(",") if t.strip()]
    first = tokens[0] if tokens else "PM2.5"
    return POLLUTANT_NAMES.get(first, f"{first} (Dominant ambient pollutant)")


def compute_forecasts(aqi_val: float, d_obj: datetime) -> list[dict]:
    horizons_meta = [
        ("Tomorrow", 1, 215.0, 0.95),
        ("Day After", 2, 230.0, 0.96),
        ("3 Days Ahead", 3, 240.0, 0.97),
    ]
    results = []
    already_bad = aqi_val > 200.0

    for name, h, cont_lvl, onset_thr in horizons_meta:
        target_d = d_obj + timedelta(days=h)
        date_label = target_d.strftime("%b %d")

        if already_bad:
            prob = max(65.0, min(96.0, 88.0 + (aqi_val - 200.0) * 0.05 - (h - 1) * 3.5))
            alert = (aqi_val >= cont_lvl)
            desc = "severe smog continuation" if aqi_val > 300 else "poor air quality continuation"
        else:
            prob = max(1.5, min(80.0, (aqi_val / 200.0) ** 2 * 50.0 + (h - 1) * 2.0))
            alert = ((prob / 100.0) >= onset_thr)
            desc = "poor air quality onset risk"

        badge = "🔴 ALERT" if alert else "🟢 SAFE"
        results.append({
            "name": name,
            "horizon": h,
            "target_date": target_d.strftime("%Y-%m-%d"),
            "date_label": date_label,
            "prob": round(prob, 1),
            "alert": bool(alert),
            "badge": badge,
            "desc": desc,
        })
    return results


def get_report(city: str, date_query: str, state_hint: str = "") -> dict:
    if not DB_PATH.exists():
        return {"error": "Database not found."}

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Query exact city & date
    cur.execute(
        "SELECT date, state, area, number_of_monitoring_stations, prominent_pollutants, aqi_value, air_quality_status, latitude, longitude FROM aqi_records WHERE area = ? AND date = ?",
        (city, date_query),
    )
    row = cur.fetchone()

    # If date not found for city, fallback to latest recorded observation
    date_fallback_note = None
    if not row:
        cur.execute(
            "SELECT date, state, area, number_of_monitoring_stations, prominent_pollutants, aqi_value, air_quality_status, latitude, longitude FROM aqi_records WHERE area = ? ORDER BY date DESC LIMIT 1",
            (city,),
        )
        row = cur.fetchone()
        if row:
            date_fallback_note = f"Date {date_query} was not in records; showing closest recorded observation from {row[0]}."

    if not row:
        conn.close()
        return {"error": f"City '{city}' not found in database."}

    conn.close()

    rec_date, rec_state, rec_area, stations, raw_pollutant, aqi_val, status, lat, lon = row
    aqi_val = float(aqi_val)
    status = str(status).strip().title() or ("Poor" if aqi_val > 200 else "Moderate")
    if status not in ADVISORIES:
        status = "Poor" if aqi_val > 200 else "Moderate"

    d_obj = datetime.strptime(rec_date[:10], "%Y-%m-%d")
    d_display = d_obj.strftime("%B %d, %Y")
    adv = ADVISORIES.get(status, ADVISORIES["Moderate"])
    forecasts = compute_forecasts(aqi_val, d_obj)

    return {
        "success": True,
        "city": rec_area,
        "state": state_hint or rec_state,
        "coordinates": {"lat": round(float(lat), 2) if lat else 0.0, "lon": round(float(lon), 2) if lon else 0.0},
        "date": d_display,
        "raw_date": rec_date,
        "aqi": int(aqi_val),
        "status": status.upper(),
        "main_hazard": parse_hazard(raw_pollutant),
        "stations": int(stations) if stations else 1,
        "date_note": date_fallback_note,
        "advisory": adv,
        "outlook": forecasts,
    }


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

        if parsed.path.endswith("/cities"):
            if CITIES_PATH.exists():
                self.wfile.write(CITIES_PATH.read_bytes())
            else:
                self.wfile.write(json.dumps({"error": "Cities registry not found."}).encode("utf-8"))
            return

        city = params.get("city", ["Patna"])[0]
        date_q = params.get("date", ["2024-11-15"])[0]
        state = params.get("state", [""])[0]

        report = get_report(city=city, date_query=date_q, state_hint=state)
        self.wfile.write(json.dumps(report, indent=2).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
