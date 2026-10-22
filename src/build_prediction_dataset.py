"""Build a clean, AQI-prediction-ready dataset.

Inputs
------
- data/processed/india_aqi_weather_daily_2022-04-01_to_2025-04-30.csv
    291 areas, AQI + weather features.
- data/raw/pollutants_source/city_day.csv
    Individual pollutant concentrations for 5 metro cities (2015-2024).

Outputs
-------
- data/processed/aqi_prediction_full.csv
    All 291 areas.  Pollutant columns are NaN for non-metro areas.
    One-hot-encoded pollutant indicators + all weather + temporal features.

- data/processed/aqi_prediction_metro.csv
    5 metro cities only, with COMPLETE pollutant + weather columns.
    Zero nulls in any prediction feature.  Use this for modelling.

- data/processed/prediction_feature_guide.json
    Lists every column and its role (target / feature-group).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
AQI_WEATHER  = ROOT / "data" / "processed" / "india_aqi_weather_daily_2022-04-01_to_2025-04-30.csv"
POLLUTANTS   = ROOT / "data" / "raw" / "pollutants_source" / "city_day.csv"
OUT          = ROOT / "data" / "processed"

CITY_TO_AREA = {
    "Delhi":     "Delhi",
    "Mumbai":    "Mumbai",
    "Chennai":   "Chennai",
    "Kolkata":   "Kolkata",
    "Bangalore": "Bengaluru",
}

POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene",
]

WEATHER_COLS = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "relative_humidity_2m_mean", "precipitation_sum",
    "wind_speed_2m_mean", "wind_speed_2m_max",
    "surface_pressure_mean", "shortwave_radiation_sum",
]

# All unique pollutant tokens that appear in prominent_pollutants
KNOWN_POLLUTANTS = ["PM2.5", "PM10", "NO2", "NH3", "CO", "SO2", "O3",
                    "NOx", "Benzene", "Toluene", "Xylene", "NO"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_temporal(df: pd.DataFrame) -> pd.DataFrame:
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek   # 0 = Monday
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    season_map = {
        12: "Winter", 1: "Winter",  2: "Winter",
         3: "Summer", 4: "Summer",  5: "Summer",
         6: "Monsoon",7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
        10: "Post-Monsoon", 11: "Post-Monsoon",
    }
    df["season"] = df["month"].map(season_map)
    return df


def encode_prominent(df: pd.DataFrame) -> pd.DataFrame:
    """
    'prominent_pollutants' looks like "PM10,PM2.5,NO2".
    Expand it into one binary column per pollutant so ML models
    can use the information without string parsing at runtime.
    """
    for p in KNOWN_POLLUTANTS:
        col = f"prominent_{p.replace('.', '').replace('-','')}"
        df[col] = df["prominent_pollutants"].str.contains(
            p.replace(".", r"\."), regex=True, na=False
        ).astype(int)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load base dataset
    # ------------------------------------------------------------------
    print("Loading AQI + weather dataset ...")
    base = pd.read_csv(AQI_WEATHER, parse_dates=["date"])
    print(f"  {len(base):,} rows, {base.area.nunique()} areas")

    # ------------------------------------------------------------------
    # 2. Load pollutant concentrations (metro cities only)
    # ------------------------------------------------------------------
    print("Loading pollutant concentrations ...")
    poll = pd.read_csv(POLLUTANTS)
    poll["date"] = pd.to_datetime(poll["Datetime"], errors="coerce")
    poll = poll.dropna(subset=["date"]).copy()
    poll["area"] = poll["City"].map(CITY_TO_AREA)
    poll = poll.dropna(subset=["area"])
    poll = poll[["date", "area"] + POLLUTANT_COLS]
    print(f"  {len(poll):,} rows, {poll.area.nunique()} metro cities")

    # ------------------------------------------------------------------
    # 3. Merge pollutants (left join keeps all 291 areas)
    # ------------------------------------------------------------------
    print("Merging ...")
    df = base.merge(poll, on=["date", "area"], how="left")

    # ------------------------------------------------------------------
    # 4. Feature engineering
    # ------------------------------------------------------------------
    df = add_temporal(df)
    df = encode_prominent(df)

    # Drop columns not useful for modelling
    DROP = ["location_key", "match_quality", "weather_source",
            "unit", "note"]
    df = df.drop(columns=[c for c in DROP if c in df.columns])

    df = df.sort_values(["date", "state", "area"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. Save FULL dataset (all 291 areas)
    # ------------------------------------------------------------------
    full_path = OUT / "aqi_prediction_full.csv"
    df.to_csv(full_path, index=False)
    print(f"\n[DONE] Full dataset  -> {full_path.name}")
    print(f"       {len(df):,} rows x {df.shape[1]} columns")
    print(f"       Rows WITH pollutant data : {df['PM2.5'].notna().sum():,}")
    print(f"       Rows WITHOUT pollutant data: {df['PM2.5'].isna().sum():,}")

    # ------------------------------------------------------------------
    # 6. Save METRO dataset (5 cities, complete pollutant data)
    # ------------------------------------------------------------------
    metro_areas = list(CITY_TO_AREA.values())
    metro = df[df["area"].isin(metro_areas) & df["PM2.5"].notna()].copy()
    metro = metro.reset_index(drop=True)
    metro_path = OUT / "aqi_prediction_metro.csv"
    metro.to_csv(metro_path, index=False)
    print(f"\n[DONE] Metro dataset -> {metro_path.name}")
    print(f"       {len(metro):,} rows x {metro.shape[1]} columns, 0 nulls in features")
    print(f"       Cities: {sorted(metro.area.unique())}")
    print(f"       Dates : {metro.date.min().date()} to {metro.date.max().date()}")

    # ------------------------------------------------------------------
    # 7. Feature guide JSON
    # ------------------------------------------------------------------
    prominent_cols = [c for c in df.columns if c.startswith("prominent_") and c != "prominent_pollutants"]
    guide = {
        "target": ["aqi_value"],
        "pollutant_concentrations": POLLUTANT_COLS,
        "pollutant_indicator_flags": prominent_cols,
        "weather_features": WEATHER_COLS,
        "temporal_features": ["year", "month", "day_of_week", "day_of_year", "is_weekend", "season"],
        "spatial_features": ["state", "area", "latitude", "longitude"],
        "metadata_only_do_not_use_as_features": [
            "air_quality_status",   # derived from aqi_value (leakage!)
            "prominent_pollutants", # raw text, replaced by indicator flags
            "number_of_monitoring_stations",
        ],
        "notes": {
            "air_quality_status": "DO NOT use as a feature - it is derived directly from aqi_value (data leakage).",
            "pollutant_concentrations": "Only available for Delhi, Mumbai, Chennai, Kolkata, Bengaluru. NaN elsewhere.",
            "pollutant_indicator_flags": "Binary flags (0/1) from prominent_pollutants field. Available for ALL 291 areas.",
            "metro_dataset": "Use aqi_prediction_metro.csv for best prediction accuracy (full feature set, no nulls).",
            "full_dataset": "Use aqi_prediction_full.csv for nationwide modelling (weather + temporal + indicator flags only).",
        }
    }
    guide_path = OUT / "prediction_feature_guide.json"
    with open(guide_path, "w", encoding="utf-8") as f:
        json.dump(guide, f, indent=2)

    # ------------------------------------------------------------------
    # 8. Print feature summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("FEATURE GUIDE FOR AQI PREDICTION")
    print("=" * 65)
    rows = [
        ("TARGET",                    guide["target"]),
        ("Pollutant concentrations",   guide["pollutant_concentrations"]),
        ("Pollutant indicator flags",  guide["pollutant_indicator_flags"]),
        ("Weather",                    guide["weather_features"]),
        ("Temporal",                   guide["temporal_features"]),
        ("Spatial",                    guide["spatial_features"]),
        ("!! DO NOT USE (leakage)",    guide["metadata_only_do_not_use_as_features"]),
    ]
    total = 0
    for label, cols in rows:
        sample = ", ".join(cols[:3]) + ("..." if len(cols) > 3 else "")
        print(f"  {label:<35} {len(cols):>3} cols  [{sample}]")
        if "DO NOT" not in label:
            total += len(cols)
    print("-" * 65)
    print(f"  {'TOTAL usable features':<35} {total:>3}")
    print(f"\nFeature guide saved -> {guide_path.name}")


if __name__ == "__main__":
    main()
