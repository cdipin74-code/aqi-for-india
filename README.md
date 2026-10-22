# India daily AQI + weather dataset

This project produces a daily, area-level AQI and weather dataset for India, and a
leakage-safe multi-horizon AQI forecasting and rare-event early-warning study built on it.

> **What this project can and cannot claim.** The experiments are a *historical back-test*.
> Weather inputs are NASA POWER reanalysis (not operational forecasts), and the AQI source is a
> Kaggle area-level aggregate whose CPCB provenance has not been independently verified.
>
> The system is deliberately split in two, because the two halves have different predictability:
> a **continuation head** for areas already above the limit (persistence is close to optimal
> there, and the paper says so), and an **onset head** for clean areas about to become hazardous
> (where persistence has exactly zero recall and the learned model detects 48.4% of one-day
> onsets at a 4.7% false-alarm rate). The combined system dominates the persistence rule at
> every false-alarm budget tested. See `paper/manuscript_draft.md` Sections 5.5–5.8.

## Research results

| Artefact | Location |
|---|---|
| Manuscript | `paper/manuscript_draft.md` |
| Project summary and corrected results | `PROJECT_SUMMARY.md` |
| Reproduction order | `RUN_ORDER.md` |
| Provenance and caveats | `SOURCES.md` |

## Live early-warning demo (5-city pilot)

The research study is a historical back-test. On top of it, `notebooks/AQI_EARLY_WARNING_FINAL_2026-10-22.ipynb`
runs the operational layer end to end and reproduces both halves in one pass:

1. the Part A research results (eight-model benchmark, horizon decay, nationwide event models); and
2. Part B, which loads official CPCB data collected daily at the 15:30 IST anchor, Open-Meteo weather,
   CPCB-vs-research parity evidence, and the current +1/+2/+3-day decisions for Delhi, Mumbai,
   Bengaluru, Chennai and Kolkata.

The executed copy with outputs and figures is
`reports/executed_notebooks/AQI_EARLY_WARNING_FINAL_2026-10-22.ipynb` (HTML alongside it). Daily
issues land in `reports/live/predictions_<date>.json`. The layer forecasts AQI > 200 (Poor or worse),
not a numeric AQI value, and covers five pilot cities while the trained model remains nationwide.

## Included AQI source

`data/raw/aqi_source/aqi.csv` is the Kaggle dataset **India Air Quality Index (AQI) Dataset [2023–2025]** by Sai Kiran Udayana (MIT licence):

<https://www.kaggle.com/datasets/saikiranudayana/india-air-quality-index-aqi-dataset-20232025>

It contains 235,785 daily AQI records for 291 areas in 32 states. Its actual available dates are **2022-04-01 to 2025-04-30**. It does *not* contain data for 2020–2021 or after April 2025.

## Weather source

Daily weather is downloaded from the [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/), using a geocoded city/area coordinate. The output therefore contains **area-level reanalysis weather**, not weather directly measured at an AQI monitor.

## Build

```powershell
python -m pip install -r requirements.txt
python src/build_dataset.py
```

The script will create:

- `data/interim/geocoded_locations.csv` — coordinates and matching status for every AQI area.
- `data/interim/weather_daily.csv` — cached weather observations.
- `data/processed/india_aqi_weather_daily_2022-04-01_to_2025-04-30.csv`
- `data/processed/coverage_report.json`

The build caches API outputs, so it is safe to rerun. Check `geocoded_locations.csv` for rows marked `review` or `unmatched`. `review` rows are intentionally excluded from the weather join until corrected. To correct a location, create `data/manual_location_overrides.csv` with these columns:

```csv
state,area,latitude,longitude
Tamil Nadu,Chennai,13.0827,80.2707
```

Then rerun the build.

## Output columns

The final file contains AQI fields (`aqi_value`, `air_quality_status`, monitoring-station count, and prominent pollutant); location fields (`state`, `area`, latitude, longitude); and daily weather fields including mean/min/max temperature, mean relative humidity, precipitation, wind, pressure, and solar radiation.

## Extending the dates

The program uses the date range present in the input AQI file. To cover 2020–2026, replace or append a legally obtained official CPCB AQI extract to `data/raw/aqi_source/aqi.csv` with the same columns. Then run:

```powershell
python src/build_dataset.py --start 2020-01-01 --end 2026-10-17
```

It will only produce rows for dates that actually exist in the AQI input; it does not invent missing AQI records.
