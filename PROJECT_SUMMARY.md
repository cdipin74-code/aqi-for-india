# Complete Project Summary & Research Documentation

**Project Title:** Nationwide Multi-Day Air Quality Index (AQI) & Extreme Smog Early Warning System for India  
**Date:** October 17, 2026  
**Primary Tooling:** Pure Machine Learning (XGBoost, Random Forest), NASA POWER Meteorology, CPCB CAAQM Ground Data  

---

## 1. Datasets Overview

Two primary datasets were structured and saved locally in `data/processed/`:

### A. Full Nationwide Dataset (`aqi_prediction_full.csv`)
* **Total Rows:** 235,785 daily observations
* **Temporal Coverage:** April 1, 2022 to April 30, 2025 (~3 years)
* **Spatial Coverage:** 291 Urban Areas across 32 States & Union Territories
* **Total Usable Features:** 44
  * **Meteorological (NASA POWER):** Mean Temperature, Min/Max Temperature, Relative Humidity, Wind Speed, Max Wind Speed, Surface Pressure, Precipitation Sum, All-Sky Solar Radiation.
  * **Geographical:** Latitude, Longitude, State, Area.
  * **Temporal / Seasonal:** Month, Day of Year, Day of Week, Is Weekend, Season (Winter, Summer, Monsoon, Post-Monsoon).
  * **Indicator Flags:** One-hot encoded prominent pollutant binary flags (`prominent_PM25`, `prominent_PM10`, `prominent_NO2`, `prominent_CO`, `prominent_SO2`, `prominent_O3`, `prominent_NH3`).
* **Target:** Continuous `aqi_value`.

### B. Metro Dataset (`aqi_prediction_metro.csv`)
* **Total Rows:** 5,027 complete daily observations (0 nulls in any feature)
* **Cities:** Delhi, Mumbai, Bengaluru, Chennai, Kolkata
* **Features Included:** All meteorological and temporal features from the full dataset PLUS exact daily chemical concentrations:
  * **Particulates & Gases:** PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene.

---

## 2. Actual Ground-Truth AQI Distribution in Your Data

Exact event distributions verified directly from your files:

| AQI Category | Numerical Range | Full Nationwide Dataset (291 Areas) | Metro Dataset (5 Hubs) | Health & Regulatory Meaning |
| :--- | :---: | :---: | :---: | :--- |
| **Good / Satisfactory** | <= 100 | 130,868 rows (55.5%) | 3,170 rows (63.1%) | Minimal impact |
| **Moderate** | 101 – 200 | 77,537 rows (32.9%) | 1,206 rows (24.0%) | Breathing discomfort to sensitive groups |
| **Poor** | 201 – 300 | 21,154 rows (9.0%) | 422 rows (8.4%) | Breathing discomfort to most upon exposure |
| **Very Poor** | 301 – 400 | 5,672 rows (2.4%) | 192 rows (3.8%) | Respiratory illness; Stage-III Alerts |
| **Severe / Hazardous** | 401 - 500+ | 554 rows (0.2%) | 37 rows (0.7%) | Serious health impact; School & Truck bans |
| **Total Health Alert (AQI > 200)** | **> 200** | **27,380 rows (11.6%)** | **651 rows (13.0%)** | **Unhealthy health warnings** |
| **Total Severe Smog (AQI > 300)** | **> 300** | **6,226 rows (2.6%)** | **229 rows (4.6%)** | **Emergency episodic smog crisis** |

### Top Regions With Highest Severe Smog (AQI > 300) Frequency:
1. **Delhi:** 240 days
2. **Byrnihat:** 197 days
3. **Greater Noida:** 170 days
4. **Noida:** 156 days
5. **Chhapra:** 150 days
6. **Bhagalpur:** 147 days
7. **Begusarai:** 144 days
8. **Saharsa:** 143 days

---

## 3. Initial Baseline Machine Learning Results

Evaluated using a strict chronological split (Train: April 2022 to June 2024; Test on unseen future: July 2024 to December 2024):

### Metro Model (Fusing Chemical Concentrations + Weather)
* **Linear Regression:** MAE = 27.27, RMSE = 39.82, R2 = 0.7702
* **Random Forest:** MAE = 20.38, RMSE = 29.35, R2 = 0.8752
* **XGBoost (Best Model):** **MAE = 20.05 AQI points**, **RMSE = 29.10**, **R2 = 0.8773**
  * **40.54%** of all predictions fell within +-10 points of true AQI.
  * Saved locally at `models/aqi_best_model.joblib`.

> **Note on `src/train_aqi_model.py`:** this older standalone trainer predicts *same-day* AQI from same-day inputs. It is a contemporaneous estimation model, not a forecast, and its R² must not be quoted as forecasting skill. The prospective multi-horizon results are in Sections 4 and 5 below.

### Non-Metro Meteorological Model (Weather + Geography + Temporal Proxies Only)
* **Random Forest / XGBoost:** R2 = 0.5037, MAE = 31.10.
* **Key Finding:** In the absence of ground-level chemical concentration monitors, reanalysis meteorology and regional seasonality explain approximately 50% of AQI variance, highlighting the need for sensor infrastructure across Tier-2/3 cities.

---

## 3b. Corrected Early-Warning Results (current)

All numbers below come from `reports/calibration/`, `reports/regional_early_warning/` and
`reports/ablation/`, generated with leakage-safe features and prior-year seasonal validation.

### Leakage-safe AQI > 200 warning, global vs regional models

Validation folds are the two prior November–April seasons (2022-11 to 2023-04, 2023-11 to 2024-04);
the threshold maximizes mean validation recall subject to FPR <= 5% in *every* fold. The test
period (2024-11-01 to 2025-04-27) is never used for fitting or selection.

| Horizon | Scope | Threshold | ROC-AUC | PR-AUC | Recall | Precision | Test FPR |
|---|---|---:|---:|---:|---:|---:|---:|
| t+1 | Global | 0.90 | 0.943 | **0.739** | 0.580 | **0.736** | **0.031** |
| t+1 | Regional | 0.83 | 0.936 | 0.713 | **0.617** | 0.679 | 0.043 |
| t+2 | Global | 0.90 | 0.917 | **0.648** | 0.471 | **0.680** | **0.033** |
| t+2 | Regional | 0.83 | 0.907 | 0.608 | **0.521** | 0.623 | 0.046 |
| t+3 | Global | 0.93 | 0.906 | **0.617** | **0.311** | **0.747** | 0.016 |
| t+3 | Regional | 0.92 | 0.895 | 0.570 | 0.245 | 0.711 | **0.015** |

**Conclusion: split-by-region training does not improve this task.** Regional models buy
2–5 points of recall at t+1/t+2 by issuing ~40% more alerts, lose 0.03–0.05 PR-AUC, and
exceed the false-alarm budget. The nationwide model is retained. This is a negative result
worth reporting rather than hiding.

Trained national models and their operating thresholds are saved at
`models/regional_early_warning/global_aqi_gt_200_h{1,2,3}.joblib` with matching
`*_threshold.json` metadata (feature lists, training cutoff, threshold).

### What actually drives warning skill (feature ablation)

Identical folds, thresholds, and test period; cumulative feature blocks.

| Horizon | Feature set | PR-AUC | Test FPR | Recall | Precision |
|---|---|---:|---:|---:|---:|
| t+1 | AQI history only (33) | 0.731 | 0.030 | 0.564 | 0.734 |
| t+1 | + weather (87) | **0.739** | 0.031 | 0.580 | 0.736 |
| t+1 | + pollutants (123) | 0.740 | 0.028 | 0.555 | 0.749 |
| t+2 | AQI history only | 0.635 | 0.031 | 0.462 | 0.685 |
| t+2 | + weather | **0.650** | 0.037 | 0.501 | 0.665 |
| t+2 | + pollutants | 0.649 | 0.038 | 0.502 | 0.663 |
| t+3 | AQI history only | 0.600 | 0.019 | 0.331 | 0.716 |
| t+3 | + weather | **0.617** | 0.015 | 0.306 | 0.755 |
| t+3 | + pollutants | 0.614 | 0.015 | 0.313 | 0.749 |

**Weather is worth ~0.01 PR-AUC at t+1 and ~0.02 at t+3. Pollutants add nothing.**
The warning task is overwhelmingly AQI persistence.

Threshold-free persistence baseline (current AQI used as a ranking score, same
threshold rule):

| Target | Horizon | Persistence PR-AUC | Model PR-AUC | Gain |
|---|---|---:|---:|---:|
| AQI > 200 | t+1 | 0.703 | 0.735 | +0.033 |
| AQI > 200 | t+2 | 0.584 | 0.645 | +0.061 |
| AQI > 200 | t+3 | 0.537 | 0.607 | +0.070 |
| AQI > 300 | t+1 | 0.483 | 0.468 | **-0.015** |
| AQI > 300 | t+2 | 0.330 | 0.329 | -0.001 |
| AQI > 300 | t+3 | 0.258 | 0.294 | +0.035 |

**For AQI > 300 the model is no better than persistence at ranking.** The 0.96
ROC-AUC for that target is a restatement of autocorrelation, not meteorological skill.
Charts: `reports/figures/pr_roc_with_baselines_*.png`.

Fixed rules on the test period (t+1):
* persistence rule "alert if AQI_t > 200": recall 0.664, precision 0.652, FPR 0.052, 5,170 alerts
* area-month climatology rule: recall 0.564, precision 0.426, FPR 0.112
* model at the **matched** 5.2% FPR: recall 0.695, precision 0.664, 5,307 alerts

So the model beats the trivial "it is already bad, warn for tomorrow" rule by
about **3 points of recall**. No-skill PR-AUC = test prevalence = 0.129 (AQI > 200)
and 0.018 (AQI > 300).

### Alert policies: what actually held on test data

The 5% false-positive target was met during tuning but **did not hold on the test period**
for AQI > 200 under any policy (test FPR 0.10–0.28). This is distribution shift between the
monsoon-dominated tuning window and the winter test window, and it is reported explicitly
via separate `tuning_fpr_target_met` and `test_fpr_target_met` columns in
`reports/calibration/calibrated_alert_policy_metrics.csv`. No AQI > 200 policy is described
as false-alarm-constrained.

For **AQI > 300**, the Balanced-F1 policy does hold below 5% test FPR at all horizons
(1.4% / 0.6% / 0.8%), but at low recall (0.57 / 0.29 / 0.27). The high-confidence
safety-first thresholds (0.26 / 0.26 / 0.16) fire rarely and are correct ~70% of the time
while missing most severe events.

### Episode-onset warning: where the ML system earns its place

Persistence fails structurally at onset: on origins with AQI(t) ≤ 200 it predicts
"no alert" for every row, so recall is exactly 0.

Test period (Nov 2024 – Apr 2025), clean origins only:

| Horizon | Score | PR-AUC | Recall | Precision | Test FPR |
|---|---|---:|---:|---:|---:|
| t+1 | XGBoost onset model | **0.354** | **48.4%** | 35.0% | 4.7% |
| t+1 | AQI level (best heuristic) | 0.270 | 34.3% | 32.7% | 3.7% |
| t+1 | Area-month climatology | 0.168 | 29.5% | 21.5% | 5.7% |
| t+1 | 1-day / 3-day / 7-day trend rules | 0.079 / 0.070 / 0.071 | ~12% | ~0.13 | ~4-5% |
| t+1 | Persistence | 0.050 | **0.0%** | — | 0.0% |
| t+2 | XGBoost onset model | **0.337** | **43.5%** | 36.1% | 5.2% |
| t+2 | AQI level | 0.238 | 25.9% | 30.7% | 3.9% |
| t+3 | XGBoost onset model | **0.351** | **30.6%** | 41.6% | 3.1% |
| t+3 | AQI level | 0.213 | 22.2% | 27.8% | 4.2% |

Onset gain over the best threshold heuristic: **+0.084 / +0.099 / +0.138 PR-AUC**.
Note the pattern reverses the continuation result: the model's edge *grows* with
lead time, because anticipating a transition needs meteorology whereas predicting
that dirty air stays dirty does not.

### Two-head system vs persistence across false-alarm budgets

Continuation head (calibrated AQI level) + onset head (calibrated probability),
selected jointly on prior winter seasons, evaluated once on the held-out period:

| Horizon | Budget | Persistence recall | Two-head recall | Persistence FPR | Two-head FPR |
|---|---:|---:|---:|---:|---:|
| t+1 | 5% | 57.6% | **61.4%** | 3.6% | 4.0% |
| t+1 | 10% | 67.1% | **81.4%** | 5.3% | 9.1% |
| t+1 | 20% | 67.1% | **93.0%** | 5.3% | 19.6% |
| t+2 | 5% | 42.0% | **42.9%** | 3.4% | 3.5% |
| t+2 | 10% | 58.2% | **67.0%** | 6.9% | 8.5% |
| t+3 | 5% | 34.4% | **35.2%** | 3.0% | 3.0% |
| t+3 | 20% | 55.1% | **81.5%** | 7.5% | 16.9% |

Persistence saturates at 67.1% (t+1) and 58.2% (t+2) because a single deterministic
threshold has no further capacity once it reaches 200. That bound is exactly what
the onset head lifts. Figure: `reports/figures/system_operating_curve.png`.

### Framing

NASA POWER is reanalysis, not forecast weather, and the AQI source is a Kaggle aggregate
whose CPCB provenance has not been independently verified. Everything above is therefore a
**historical back-test**, not an operational live forecast.

The headline is now a separation of concerns rather than a single accuracy claim:

* **Continuation days** (already hazardous): persistence is close to optimal, and the
  ML models beat it only modestly. Report this, do not hide it.
* **Onset days** (clean becoming hazardous): persistence has 0% recall, and the ML
  onset model is the only component with signal. This is where a warning system
  actually earns its existence.
* **System level**: the two-head combination dominates persistence at every
  false-alarm budget tested, from 2% to 20%.



---

## 4. The Research Paper Blueprint

### Working Title:
> **"Episode-Onset Aware Multi-Day AQI Early Warning for India: A Leakage-Safe Machine Learning System Evaluated Against Persistence"**

### Core Contributions / Novelty:
1. **Episode-Onset Formulation:** Separating *continuation* (already hazardous) from *onset* (clean becoming hazardous), and showing that persistence — which solves most of the first — has exactly zero recall on the second.
2. **A Warning System That Beats Persistence Everywhere:** A two-head architecture whose combined alert stream dominates the persistence rule at false-alarm budgets from 2% to 20%, with the mechanism made explicit (a deterministic rule is bounded at its threshold).
3. **Multi-Horizon Forecasting (t+1, t+2, t+3 Days):** Leakage-safe features only, evaluated against persistence and seasonal climatology at every horizon.
4. **Nationwide Scale:** 291 areas in 32 states, with regional partition models tested and rejected as a negative result.
5. **Policy Relevance:** Calibrated operating points linked to health-risk transitions and India's statutory Graded Response Action Plan (GRAP stages I–IV), with honest out-of-sample false-alarm rates.

---

## 4b. Live Pilot Deployment (post-study)

The research results above are a historical back-test. A separate operational layer extends the
same trained models to official CPCB data collected daily from 2025-05-01 for five pilot cities
(Delhi, Mumbai, Bengaluru, Chennai, Kolkata), with Open-Meteo weather.

- **Validation against the research panel** (2025-01-01 to 2025-04-30, 117–119 common days per
  city-anchor): MAE 0.89–3.19 AQI points, median absolute error 1–2, r 0.979–0.999, 85–99% of
  days within 5 points. Weather matches NASA POWER after unit alignment, except relative
  humidity, which retains a source offset of up to +17.6 points in Delhi.
- **Daily decision:** AQI > 200 at +1/+2/+3 days, using the calibrated continuation/onset pair
  (215/0.95, 230/0.96, 240/0.97). Outputs are written to `reports/live/predictions_<date>.json`.
- **Single reproducible artefact:** `notebooks/AQI_EARLY_WARNING_FINAL_2026-10-22.ipynb`
  reproduces Part A (research) and Part B (live layer) in one run.

---

## 5. Local File Locations on Your Computer

* **Metro Dataset (Prediction-Ready):**  
  `data/processed/aqi_prediction_metro.csv`
* **Full Nationwide Dataset (291 Areas):**  
  `data/processed/aqi_prediction_full.csv`
* **Feature Reference & Metadata Guide:**  
  `data/processed/prediction_feature_guide.json`
* **Trained Best Model:**  
  `models/aqi_best_model.joblib`
* **Feature Importance Plot:**  
  `reports/feature_importance.png`
* **Model Comparison Plot:**  
  `reports/model_comparison.png`
* **Actual vs. Predicted Plot:**  
  `reports/actual_vs_predicted.png`
* **Test Predictions CSV:**  
  `reports/predictions_test.csv`
* **Full Summary Report JSON:**  
  `reports/model_report.json`
