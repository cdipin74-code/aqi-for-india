"""Compute each pilot's k-nearest research areas and match them to CPCB city ids.

The onset models were trained with add_neighbour_features(k=5): each area's
neighbour mean/max AQI come from the 5 geographically closest areas that report
in the research frame.  For live dates the frame only spans the CPCB city-level
feeds, so the plan below records, for every pilot, exactly which research areas
act as neighbours, and the matching CPCB city master entry used to fetch them.

Writes data/raw/cpcb_live/neighbour_plan.json with:
  pilots[name] = {location_key, ctpb_city_id, neighbours: [{area, location_key,
                   ctpb_city_id(cpcb), match, distance_km}...] (5, nearest first,
                   matched to CPCB only if ctpb_city_id is non-empty)}
"""
from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LIVE_DIR = ROOT / "data" / "raw" / "cpcb_live"
GEO = ROOT / "data" / "interim" / "geocoded_locations.csv"
PILOTS = {
    "Delhi": "CTY-000170",
    "Mumbai": "CTY-000396",
    "Bengaluru": "CTY-000081",
    "Chennai": "CTY-000134",
    "Kolkata": "CTY-000320",
}
K = 5  # onset NEIGHBOURS
R_EARTH = 6371.0


def haversine(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    la1, lo1 = np.radians(lat1), np.radians(lon1)
    la2, lo2 = np.radians(lat2), np.radians(lon2)
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def match_cpcb_city(area: str, master: dict[str, dict]) -> tuple[str, str, float]:
    """Match a research area name to the CPCB city master.

    Returns (city_id, city_name, score). Normalise 'bengaluru'/'bangalore'.
    """
    canon = {
        "bangaore": "bengaluru", "bangalore": "bengaluru", "calcutta": "kolkata",
        "madras": "chennai", "bombay": "mumbai", "gurgaon": "gurugram",
        "na": "", "nct": "",
    }
    norm = area.strip().lower().strip()

    def clean(x: str) -> str:
        x = x.lower().strip()
        for bad in ("(", ")", "cg", "mc", "biz", "-", ","):
            x = x.replace(bad, " ")
        return " ".join(x.split())

    a = canon.get(norm, clean(norm))
    if a == "bengaluru" and "banga" in a:  # keep simple
        pass
    a = canon.get(norm, norm).replace("bangalore", "bengaluru")
    best = ("", "", 0.0)
    for cid, row in master.items():
        cname = clean(row["city_name"])
        cname = cname.replace("bangalore", "bengaluru")
        if cname == a:
            return cid, row["city_name"], 1.0
        score = difflib.SequenceMatcher(None, cname, a).ratio()
        if score > best[2]:
            best = (cid, row["city_name"], score)
    if best[2] >= 0.74:
        return best
    return "", "", best[2]


def main() -> None:
    geo = pd.read_csv(GEO)
    geo["area_norm"] = geo["area"].astype(str).str.strip()
    master_rows = json.loads((LIVE_DIR / "city_master.json").read_text(encoding="utf8"))
    master = {}
    for r in master_rows:
        cid, cname, sname = r if isinstance(r, list) else (r["city_id"], r["city_name"], r.get("state_name", ""))
        master[cid] = {"city_name": cname, "state_name": sname}

    plan: dict[str, dict] = {}
    report: list[str] = []
    for pilot, pilot_city_id in PILOTS.items():
        crow = geo[geo["area_norm"].str.casefold() == pilot.casefold()]
        if crow.empty:
            # research may store the pilot under another spelling
            suggest = [a for a in geo["area_norm"] if pilot.lower()[:4] == a.lower()[:4]][:5]
            report.append(f"!! pilot '{pilot}' NOT in research frame; candidates: {suggest}")
            continue
        plat, plon, pkey, pstate = crow.iloc[0][["latitude", "longitude", "location_key", "state"]]
        dist = haversine(float(plat), float(plon), geo["latitude"].to_numpy(), geo["longitude"].to_numpy())
        idx = np.argsort(dist)[: K + 6]  # a few extra in case some don't match CPCB
        neighbours = []
        for gi in idx:
            row = geo.iloc[gi]
            if row["location_key"] == pkey:
                continue
            if len(neighbours) >= K:
                break
            cid, cname, score = match_cpcb_city(str(row["area_norm"]), master)
            neighbours.append(
                {
                    "area": str(row["area_norm"]),
                    "location_key": row["location_key"],
                    "state": row["state"],
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "distance_km": round(float(dist[gi]), 1),
                    "cpcb_city_id": cid,
                    "cpcb_city_name": cname,
                    "match_score": round(score, 3),
                }
            )
        matched = sum(1 for n in neighbours if n["cpcb_city_id"])
        plan[pilot] = {
            "pilot_location_key": pkey,
            "state": str(pstate),
            "latitude": float(plat),
            "longitude": float(plon),
            "cpcb_city_id": pilot_city_id,
            "neighbours": neighbours,
            "matched_neighbours": matched,
        }
        nlist = ", ".join(
            (n["cpcb_city_name"] + " (" + n["area"] + ", " + str(n["distance_km"]) + "km, id=" + (n["cpcb_city_id"] or "-") + ")")
            if n["cpcb_city_id"]
            else n["area"] + " (" + str(n["distance_km"]) + "km, UNMATCHED)"
            for n in neighbours
        )
        report.append(f"{pilot}: {matched}/5 matched | {nlist}")

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    (LIVE_DIR / "neighbour_plan.json").write_text(json.dumps(plan, indent=1), encoding="utf8")
    out = "\n".join(report)
    (LIVE_DIR / "neighbour_plan.txt").write_text(out, encoding="utf8")
    print(out.encode("ascii", "backslashreplace").decode())


if __name__ == "__main__":
    main()
