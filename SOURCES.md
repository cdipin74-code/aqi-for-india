# Data provenance

## AQI input

- **Dataset:** India Air Quality Index (AQI) Dataset [2023–2025]
- **Publisher:** Sai Kiran Udayana, Kaggle
- **Licence:** MIT
- **URL:** https://www.kaggle.com/datasets/saikiranudayana/india-air-quality-index-aqi-dataset-20232025
- **Verified contents used here:** 235,785 daily observations, 32 states, 291 areas, 2022-04-01 through 2025-04-30.

The source reports an AQI aggregate for an `area` and date, with its number of contributing monitoring stations. It is not station-level pollutant concentration data.

## Weather enrichment

- **Provider:** NASA POWER Daily API
- **URL:** https://power.larc.nasa.gov/docs/services/api/temporal/daily/
- **Method:** One Local Solar Time daily weather series is queried for the coordinate assigned to each AQI area.

Weather is reanalysis/grid-cell data. It should be treated as local area-level meteorology, rather than as an observation recorded by the AQI instrument.

## Live operational layer (5-city pilot)

Added after the research study, and kept separate from the research inputs above.

- **AQI:** CPCB Central Control Room, `get-fn-city-aqi-summary-by-time` (encrypted API), city AQI at the 15:30 IST anchor with 17:30 IST fallback. Cached per city under `data/raw/cpcb_live/city_daily/`. Protocol documented in `docs/cpcb_recon.md`.
- **Weather:** Open-Meteo ERA5 archive plus the forecast API for recent days, mapped onto the NASA POWER feature names. Wind is requested in m/s and converted 10 m to 2 m with `(2/10)^(1/7)`; shortwave radiation is converted from MJ/m2/day to the NASA POWER unit of kWh/m2/day.
- **Scope:** five pilot cities (Delhi, Mumbai, Bengaluru, Chennai, Kolkata) from 2025-05-01 onward, plus 20 neighbouring cities over a recent window for the neighbour features. The trained model remains nationwide.
- **Parity check:** CPCB live vs the Kaggle research series over 2025-01-01 to 2025-04-30, 117–119 common days per city-anchor: MAE 0.89–3.19 AQI points, median absolute error 1–2, Pearson r 0.979–0.999, 85–99% of days within 5 points. Full table in `data/raw/cpcb_live/overlap_validation.json`.
- **Weather parity:** Open-Meteo vs NASA POWER over the same window: wind MAE 0.44–0.52 m/s with bias −0.51 to −0.33 m/s; shortwave MAE 0.36–0.62 kWh/m2/day with bias −0.02 to +0.53. Relative humidity still carries a source offset, up to +17.6 points for Delhi. Full table in `data/raw/cpcb_live/weather_overlap_validation.json`.

## Coverage caveat

There is no verified, ready-to-download nationwide public AQI table in this workspace for every date from 2020-01-01 through 2026-10-17. The final CSV's actual date coverage is stated in `data/processed/coverage_report.json`; no AQI values are interpolated or fabricated.
