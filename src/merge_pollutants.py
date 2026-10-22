"""Merge individual pollutant concentrations into the AQI + weather dataset.

This creates a prediction-ready dataset by combining:
  - AQI + weather data  (291 areas, 2022-04 to 2025-04)
  - Pollutant concentrations from city_day.csv (5 metro cities, 2015-2024)

The pollutant data only covers 5 cities (Delhi, Mumbai, Chennai, Kolkata,
Bangalore/Bengaluru), so:
  - A "full" dataset is produced with ALL 291 areas (pollutant cols are NaN
    for non-metro areas) — useful for weather-only AQI modelling.
  - A "metro" dataset is produced with only the 5 cities that have pollutant
    data — useful for pollutant-based AQI prediction models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AQI_WEATHER = ROOT / "data" / "processed" / "india_aqi_weather_daily_2022-04-01_to_2025-04-30.csv"
POLLUTANTS = ROOT / "data" / "raw" / "pollutants_source" / "city_day.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

# Mapping: pollutant-file city name -> AQI dataset area name
CITY_TO_AREA = {
    "Delhi": "Delhi",
    "Mumbai": "Mumbai",
    "Chennai": "Chennai",
    "Kolkata": "Kolkata",
    "Bangalore": "Bengaluru",
}

POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]


def load_aqi_weather() -> pd.DataFrame:
    """Load the existing AQI + weather merged dataset."""
    print("Loading AQI + weather dataset …")
    df = pd.read_csv(AQI_WEATHER, parse_dates=["date"])
    print(f"  {len(df):,} rows, {df.area.nunique()} areas, "
          f"{df.date.min().date()} to {df.date.max().date()}")
    return df


def load_pollutants() -> pd.DataFrame:
    """Load and clean the city-level pollutant concentrations."""
    print("Loading pollutant concentrations …")
    df = pd.read_csv(POLLUTANTS)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime", "City"]).copy()

    # Map city names to match AQI dataset area names
    df["area"] = df["City"].map(CITY_TO_AREA)
    df = df.dropna(subset=["area"])
    df = df.rename(columns={"Datetime": "date"})

    # Keep only pollutant concentration columns
    keep = ["date", "area"] + POLLUTANT_COLS
    df = df[keep].copy()

    print(f"  {len(df):,} rows, {df.area.nunique()} cities, "
          f"{df.date.min().date()} to {df.date.max().date()}")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features useful for AQI prediction."""
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Monday
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Season (India): Winter=Dec-Feb, Summer=Mar-May, Monsoon=Jun-Sep, Post-Monsoon=Oct-Nov
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Summer", 4: "Summer", 5: "Summer",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
        10: "Post-Monsoon", 11: "Post-Monsoon",
    }
    df["season"] = df["month"].map(season_map)
    return df


def main() -> None:
    aqi_weather = load_aqi_weather()
    pollutants = load_pollutants()

    # --- Merge pollutants into the full dataset ---
    print("\nMerging pollutant concentrations …")
    merged = aqi_weather.merge(pollutants, on=["date", "area"], how="left")

    # Add temporal features
    merged = add_temporal_features(merged)

    # Sort cleanly
    merged = merged.sort_values(["date", "state", "area"]).reset_index(drop=True)

    # --- Full dataset (all 291 areas) ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_path = OUTPUT_DIR / "india_aqi_weather_pollutants_full.csv"
    merged.to_csv(full_path, index=False)
    print(f"\n[OK] Full dataset: {full_path.name}")
    print(f"   {len(merged):,} rows, {merged.area.nunique()} areas")

    # --- Metro-only dataset (5 cities with pollutant data) ---
    metro_areas = list(CITY_TO_AREA.values())
    metro = merged[merged["area"].isin(metro_areas)].copy()

    # Within metro, further filter to the date overlap where pollutants exist
    metro_with_poll = metro[metro[POLLUTANT_COLS[0]].notna()].copy()
    metro_with_poll = metro_with_poll.reset_index(drop=True)

    metro_path = OUTPUT_DIR / "india_aqi_weather_pollutants_metro.csv"
    metro_with_poll.to_csv(metro_path, index=False)
    print(f"\n[OK] Metro dataset (prediction-ready): {metro_path.name}")
    print(f"   {len(metro_with_poll):,} rows, {metro_with_poll.area.nunique()} cities")
    print(f"   Date range: {metro_with_poll.date.min().date()} to {metro_with_poll.date.max().date()}")

    # --- Coverage report ---
    weather_fields = [
        "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        "relative_humidity_2m_mean", "precipitation_sum", "wind_speed_2m_mean",
        "wind_speed_2m_max", "surface_pressure_mean", "shortwave_radiation_sum",
    ]
    report = {
        "full_dataset": {
            "file": full_path.name,
            "total_rows": int(len(merged)),
            "areas": int(merged.area.nunique()),
            "states": int(merged.state.nunique()),
            "date_start": str(merged.date.min().date()),
            "date_end": str(merged.date.max().date()),
            "rows_with_pollutants": int(merged[POLLUTANT_COLS[0]].notna().sum()),
            "rows_with_weather": int(merged[weather_fields].notna().all(axis=1).sum()),
        },
        "metro_dataset": {
            "file": metro_path.name,
            "total_rows": int(len(metro_with_poll)),
            "cities": metro_areas,
            "date_start": str(metro_with_poll.date.min().date()),
            "date_end": str(metro_with_poll.date.max().date()),
            "rows_with_complete_data": int(
                metro_with_poll[POLLUTANT_COLS + weather_fields].notna().all(axis=1).sum()
            ),
        },
        "feature_groups": {
            "target": ["aqi_value"],
            "pollutant_features": POLLUTANT_COLS,
            "weather_features": weather_fields,
            "temporal_features": ["year", "month", "day_of_week", "day_of_year", "is_weekend", "season"],
            "spatial_features": ["state", "area", "latitude", "longitude"],
            "metadata": ["air_quality_status", "prominent_pollutants", "number_of_monitoring_stations"],
        },
    }
    report_path = OUTPUT_DIR / "merged_coverage_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] Coverage report: {report_path.name}")

    # Print a quick summary table
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY FOR AQI PREDICTION")
    print("=" * 60)
    print(f"{'Feature Group':<25} {'Count':<8} {'Columns'}")
    print("-" * 60)
    for group, cols in report["feature_groups"].items():
        print(f"{group:<25} {len(cols):<8} {', '.join(cols[:4])}{'…' if len(cols) > 4 else ''}")
    print("-" * 60)
    print(f"{'TOTAL features':<25} {sum(len(v) for v in report['feature_groups'].values())}")
    print()
    print("Recommended for prediction: Use the METRO dataset")
    print(f"  -> {metro_path.name}")
    print(f"  -> {len(metro_with_poll):,} rows with complete pollutant + weather data")


if __name__ == "__main__":
    main()
