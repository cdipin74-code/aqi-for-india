"""Feature-set ablation and no-skill baselines for the AQI >200 warning task.

The question this answers: how much of the warning skill comes from AQI
persistence alone, how much weather adds, and how much sparse pollutant data
adds on top of that.  All three feature sets use only information available at
issue date ``t``, and all use the same seasonal rolling validation folds and
threshold-selection rule as the main regional experiment.

Feature sets
------------
``AQI history only``   current + lagged/rolling AQI, calendar, and location.
``AQI + weather``      adds lagged and rolling NASA POWER reanalysis weather.
``AQI + weather + pollutants`` adds lagged and rolling pollutant concentrations.

No-skill references
-------------------
* ``prevalence``  - PR-AUC of a random score; equals the event rate.
* ``persistence`` - alert at t+h iff AQI at t already exceeds the cutoff.
* ``climatology`` - alert at t+h iff the location's historical event rate in the
  same calendar month exceeds 50%.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from train_regional_early_warning import (
    CUTOFF,
    DATA_PATH,
    HORIZONS,
    MAX_FALSE_POSITIVE_RATE,
    ROOT,
    WEATHER,
    build_feature_table,
    global_probabilities,
    metrics,
    select_robust_threshold,
    valid_shift,
)

REPORT_DIR = ROOT / "reports" / "ablation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

POLLUTANTS = (
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene",
)

VALIDATION_FOLDS = [
    ("2022-11_to_2023-04", pd.Timestamp("2022-10-31"), pd.Timestamp("2022-11-01"), pd.Timestamp("2023-04-30")),
    ("2023-11_to_2024-04", pd.Timestamp("2023-10-31"), pd.Timestamp("2023-11-01"), pd.Timestamp("2024-04-30")),
]
FINAL_TRAIN_END = pd.Timestamp("2024-10-31")
TEST_START = pd.Timestamp("2024-11-01")
TEST_END = pd.Timestamp("2025-04-27")


def add_pollutant_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add lagged/rolling pollutant values; NaN when a monitor did not report."""
    data = frame.copy()
    grouped = data.groupby("location_key", sort=False)
    derived: dict[str, pd.Series] = {}
    pollutants = [column for column in POLLUTANTS if column in data.columns]
    for column in pollutants:
        for lag in (1, 7):
            derived[f"{column}_lag_{lag}"] = valid_shift(data, grouped, column, lag)
        prior = grouped[column].shift(1)
        first_date = grouped["date"].shift(7)
        contiguous = (data["date"] - first_date).dt.days.eq(7)
        derived[f"{column}_mean_7d"] = prior.groupby(data["location_key"]).transform(
            lambda values: values.rolling(7, min_periods=7).mean()
        ).where(contiguous)
    data = pd.concat([data, pd.DataFrame(derived, index=data.index)], axis=1)
    return data, list(derived)


def split_feature_blocks(numeric: list[str], derived_pollutants: list[str]) -> dict[str, list[str]]:
    """Partition the numeric columns into cumulative, leakage-safe blocks."""
    pollutant_set = set(derived_pollutants)
    weather_roots = tuple(f"{column}_" for column in WEATHER)

    core = [column for column in numeric if column not in pollutant_set and not column.startswith(weather_roots)]
    weather = [column for column in numeric if column.startswith(weather_roots)]
    return {
        "AQI history only": core,
        "AQI + weather": core + weather,
        "AQI + weather + pollutants": core + weather + derived_pollutants,
    }


def choose_threshold(validation_predictions: pd.DataFrame) -> tuple[float, bool]:
    """Apply the FPR-in-every-fold rule, falling back to the closest feasible point."""
    try:
        threshold, _ = select_robust_threshold(validation_predictions)
        return threshold, True
    except RuntimeError:
        rows = []
        for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
            fold_fpr = [
                metrics(fold.event_observed.to_numpy(), fold.probability.to_numpy(), threshold)["false_positive_rate"]
                for _, fold in validation_predictions.groupby("fold")
            ]
            rows.append((threshold, max(fold_fpr)))
        curve = pd.DataFrame(rows, columns=["threshold", "worst_fold_fpr"])
        best = curve.sort_values(["worst_fold_fpr", "threshold"], ascending=[True, False]).iloc[0]
        return float(best.threshold), False


def persistence_rule(usable: pd.DataFrame, horizon: int) -> dict:
    """Alert for t+h whenever AQI at t already exceeds the cutoff."""
    issued = (usable["aqi_at_issue"] > CUTOFF).to_numpy()
    observed = usable["label"].to_numpy()
    tn = int(((~issued) & (observed == 0)).sum())
    fp = int((issued & (observed == 0)).sum())
    fn = int(((~issued) & (observed == 1)).sum())
    tp = int((issued & (observed == 1)).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    return {
        "horizon_days": horizon, "rule": "persistence (AQI at t > cutoff)",
        "alerts_issued": int(issued.sum()), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Recall": recall, "Precision": precision, "F1": (
            2 * precision * recall / (precision + recall) if precision and recall else 0.0
        ),
        "false_positive_rate": fpr,
    }


def climatology_rule(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Alert when the location's seasonal event rate learned on training data exceeds 50%."""
    train = train.assign(forecast_month=(train["date"] + pd.Timedelta(days=1)).dt.month)
    test = test.assign(forecast_month=(test["date"] + pd.Timedelta(days=1)).dt.month)
    rates = train.groupby(["location_key", "forecast_month"]).label.mean().rename("rate").reset_index()
    merged = test.merge(rates, on=["location_key", "forecast_month"], how="left")
    fallback = train.label.mean()
    issued = merged.rate.fillna(fallback).gt(0.5).to_numpy()
    observed = merged.label.to_numpy()
    tn = int(((~issued) & (observed == 0)).sum())
    fp = int((issued & (observed == 0)).sum())
    fn = int(((~issued) & (observed == 1)).sum())
    tp = int((issued & (observed == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    return {
        "horizon_days": 1, "rule": "climatology (seasonal event rate > 50%)",
        "alerts_issued": int(issued.sum()), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Recall": recall, "Precision": precision,
        "F1": 2 * precision * recall / (precision + recall) if precision and recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
    }


def run() -> None:
    frame, numeric, categorical = build_feature_table()
    frame, derived_pollutants = add_pollutant_features(frame)
    blocks = split_feature_blocks(numeric, derived_pollutants)
    for name, columns in blocks.items():
        print(f"{name}: {len(columns)} numeric features")

    rows, curves, predictions_out = [], [], []
    for horizon in HORIZONS:
        label = f"event_gt_{CUTOFF}_h{horizon}"
        usable = frame[frame[label].notna()].copy().rename(columns={label: "label"})
        usable["label"] = usable["label"].astype(int)

        train = usable[usable.date <= FINAL_TRAIN_END]
        test = usable[(usable.date >= TEST_START) & (usable.date <= TEST_END)].copy()

        for set_name, columns in blocks.items():
            print(f"  [{horizon}] {set_name}")
            validation_predictions = []
            for fold_name, train_end, valid_start, valid_end in VALIDATION_FOLDS:
                fold_train = usable[usable.date <= train_end]
                fold_valid = usable[(usable.date >= valid_start) & (usable.date <= valid_end)]
                probability = global_probabilities(fold_train, fold_valid, columns, categorical)
                validation_predictions.append(pd.DataFrame({
                    "fold": fold_name, "event_observed": fold_valid.label.to_numpy(), "probability": probability,
                }))
            validation_predictions = pd.concat(validation_predictions, ignore_index=True)
            threshold, feasible = choose_threshold(validation_predictions)

            probability = global_probabilities(train, test, columns, categorical)
            rows.append({
                "horizon_days": horizon, "feature_set": set_name, "n_numeric_features": len(columns),
                "threshold": threshold, "fpr_constraint_feasible": feasible,
                **metrics(test.label.to_numpy(), probability, threshold),
                "test_prevalence": float(test.label.mean()),
            })
            prediction = test[["date", "state", "area", "region", "location_key"]].copy()
            prediction["forecast_date"] = prediction.date + pd.Timedelta(days=horizon)
            prediction["horizon_days"] = horizon
            prediction["feature_set"] = set_name
            prediction["event_observed"] = test.label.to_numpy()
            prediction["probability"] = probability
            prediction["threshold"] = threshold
            prediction["alert_issued"] = (probability >= threshold).astype(int)
            predictions_out.append(prediction)

        rows.append({
            "horizon_days": horizon, "feature_set": "no skill (prevalence)", "n_numeric_features": 0,
            "threshold": np.nan, "fpr_constraint_feasible": np.nan,
            "ROC_AUC": 0.5, "PR_AUC": float(test.label.mean()),
            "Recall": np.nan, "Precision": np.nan, "false_positive_rate": np.nan,
            "TP": 0, "FP": 0, "FN": 0, "TN": 0,
            "test_prevalence": float(test.label.mean()),
        })

    ablation = pd.DataFrame(rows).sort_values(["horizon_days", "PR_AUC"], ascending=[True, False])
    ablation.to_csv(REPORT_DIR / "feature_set_ablation.csv", index=False)
    pd.concat(predictions_out, ignore_index=True).to_csv(
        REPORT_DIR / "feature_set_ablation_test_predictions.csv", index=False
    )

    # Rule-based references on the same held-out period, t+1.
    label_h1 = f"event_gt_{CUTOFF}_h1"
    usable_h1 = frame[frame[label_h1].notna()].copy().rename(columns={label_h1: "label"})
    usable_h1["label"] = usable_h1["label"].astype(int)
    train_h1 = usable_h1[usable_h1.date <= FINAL_TRAIN_END]
    test_h1 = usable_h1[(usable_h1.date >= TEST_START) & (usable_h1.date <= TEST_END)].copy()
    rule_rows = [persistence_rule(test_h1, 1), climatology_rule(train_h1, test_h1)]
    pd.DataFrame(rule_rows).to_csv(REPORT_DIR / "rule_baseline_metrics.csv", index=False)

    with (REPORT_DIR / "ablation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "input": str(DATA_PATH.relative_to(ROOT)),
            "cutoff": CUTOFF,
            "horizons": list(HORIZONS),
            "feature_sets": {name: len(columns) for name, columns in blocks.items()},
            "validation_folds": [fold[0] for fold in VALIDATION_FOLDS],
            "final_train_end": str(FINAL_TRAIN_END.date()),
            "final_test_period": f"{TEST_START.date()} to {TEST_END.date()}",
            "max_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
            "threshold_rule": "max mean recall subject to FPR <= target in every validation fold",
        }, handle, indent=2)

    print("\nAblation (held-out November 2024 - April 2025):")
    print(ablation.round(3).to_string(index=False))
    print("\nRule-based baselines:")
    print(pd.DataFrame(rule_rows).round(3).to_string(index=False))
    print(f"\nSaved to {REPORT_DIR}")


if __name__ == "__main__":
    run()
