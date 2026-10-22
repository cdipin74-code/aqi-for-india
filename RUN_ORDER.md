# Research evaluation run order

## Notebooks

Run these in Jupyter, one at a time, using **Kernel → Restart & Clear Output → Cell → Run All**.

1. `notebooks/RUN_THIS_AQI_RESEARCH.ipynb`
   - Eight-model benchmark, multi-horizon forecast, and initial cost-sensitive warning model.
2. `notebooks/STEP_1_BASELINES_AND_EVALUATION.ipynb`
   - Persistence, seasonal, and unweighted-classifier baselines.
3. `notebooks/STEP_2_CALIBRATION_AND_ALERTS.ipynb`
   - Platt calibration, reliability diagrams, and practical alert policies.
   - Reports tuning-time and out-of-sample false-positive rates separately; the
     5% FPR target is a tuning constraint, not a guarantee.
4. `notebooks/STEP_3_ROBUSTNESS_AND_CONFIDENCE_INTERVALS.ipynb`
   - State/region/season/high-risk-city metrics and date-block bootstrap confidence intervals.

## Scripts

After the notebooks, from the project root:

```powershell
# Nationwide vs regional AQI >200 warning models with prior-year seasonal folds.
# Writes metrics, predictions, threshold curves, and saved models + thresholds.
python src\train_regional_early_warning.py

# Cumulative feature ablation (AQI only / + weather / + pollutants) and rule baselines.
python src\run_feature_ablation.py

# Current AQI used as a ranking score, for a threshold-free persistence comparison.
python src\score_persistence_baseline.py

# PR/ROC curves with persistence and prevalence reference lines.
python src\plot_pr_with_baselines.py

# Persistence row for the eight-model metro benchmark (Table 1).
python src\benchmark_persistence_reference.py

# Episode-onset warning (the task persistence cannot do) and its baselines.
python src\evaluate_onset_warning.py

# Joint calibration of the two-head system (continuation + onset).
python src\calibrate_combined_system.py

# Recall-versus-budget operating curve: two-head vs persistence.
python src\system_operating_curve.py
```

`src\diagnose_weather_headroom.py` is an optional upper-bound diagnostic: it adds
*oracle* reanalysis weather from the target date, which is not deployable, to measure
how much of the remaining error is meteorological. Do not report its scores as system
performance.

Runtime is roughly 10-15 minutes for each of the model-training scripts; the onset
experiment and the system calibration add another 10-15 minutes. Everything is
deterministic under `random_state=42`.

### Output locations

| Path | Contents |
|---|---|
| `reports/research_ml/` | Benchmark, multi-horizon, and uncalibrated warning results |
| `reports/calibration/` | Calibrated policies, reliability tables, FPR/recall policy plots |
| `reports/robustness/` | Subgroup metrics and bootstrap intervals |
| `reports/regional_early_warning/` | Global vs regional comparison, threshold curves |
| `reports/ablation/` | Feature ablation, rule and persistence baselines, PR-vs-persistence table |
| `reports/onset/` | Episode-onset metrics, joint system calibration, operating curve |
| `reports/figures/` | PR/ROC curves with no-skill references; system operating curve |
| `models/regional_early_warning/` | Saved nationwide warning pipelines and their thresholds |

The manuscript template is `paper/manuscript_draft.md`. Fill its result tables only from generated CSV files after all steps finish successfully.

## Live-data pipeline (operational extension, 2025-05-01 onward)

The research dataset ends at 2025-04-30. This extension keeps the trained models
and the leakage discipline but feeds them **official CPCB CCR** daily city AQI
(retrieved through the portal's own encrypted transport; see
`docs/cpcb_recon.md`) and Open-Meteo weather for the gap. New files are written
in parallel; the research dataset is never modified.

```powershell
# 1. Live weather for the gap (Open-Meteo ERA5 archive + forecast API).
python src\fetch_weather_openmeteo.py

# 2. Official CPCB daily city AQI for the pilot cities, both IST anchors,
#    with resume cache. Re-mints its session token automatically on expiry.
python src\fetch_cpcb_live.py --start 2025-05-01 --end <today-1>

# 3. Neighbour plan: each pilot's k=5 nearest research areas matched to CPCB
#    city ids (used by the onset models' neighbour features).
python src\plan_neighbour_cities.py
python src\fetch_cpcb_live.py --cities <plan neighbours> --start <asof-40d> --end <today-1>

# 4. Live validation BEFORE parity claims: overlap fetch Jan-Apr 2025 vs the
#    Kaggle research file + weather-source comparison.
python src\validate_cpcb_overlap.py

# 5. Build the live feature frame (research rows += live rows += live weather).
python src\build_live_dataset.py

# 6. Issue next-1/2/3-day warnings for the pilots with the saved models and the
#    jointly calibrated two-head rule. JSON goes to reports/live/.
python src\predict_latest.py
```

Honest limitations of the live path (also stated in the prediction JSON):

- Weather is Open-Meteo (ERA5 archive; forecast API for recent days) — different
 source than the research reanalysis, validated on the overlap by step 4.
- Onset neighbour features use only the pilot+neighbour cities present in the
 CPCB city master; research areas without a CPCB city feed contribute nothing
 that day (e.g. Chennai: 2 of 5 planned neighbours match).
- Station-support is not available for historic days (only the last 3 days
 carry station counts through `get-city-wise-last-3-days-aqi`).

| Path | Contents |
|---|---|
| `data/raw/cpcb_live/` | Fetch caches, city master, token file, manifests, logs |
| `data/raw/cpcb_live/overlap_cache/` | Jan-Apr 2025 parity fetch (kept separate) |
| `data/processed/live_feature_frame.csv` | Research + live feature frame (supplement) |
| `reports/live/` | Daily prediction JSONs |
| `src/predict_latest.py` | Runs the two-head rule on the latest live day |
