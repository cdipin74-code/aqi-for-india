"""Leakage-safe, region-aware AQI >200 early-warning evaluation.

The script uses only information available at issue date t: current AQI,
historical AQI/weather features, location metadata, and known calendar values.
It compares a nationwide model with independently trained regional models using
rolling validation periods before evaluating once on later, untouched dates.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "aqi_prediction_full.csv"
REPORT_DIR = ROOT / "reports" / "regional_early_warning"
MODEL_DIR = ROOT / "models" / "regional_early_warning"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = (1, 2, 3)
CUTOFF = 200
RANDOM_STATE = 42
MAX_FALSE_POSITIVE_RATE = 0.05
WEATHER = (
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "relative_humidity_2m_mean", "precipitation_sum", "wind_speed_2m_mean",
    "wind_speed_2m_max", "surface_pressure_mean", "shortwave_radiation_sum",
)

STATE_REGIONS = {
    "North": {"Chandigarh", "Delhi", "Haryana", "Himachal Pradesh", "Jammu and Kashmir", "Punjab", "Rajasthan", "Uttar Pradesh", "Uttarakhand"},
    "Central": {"Chhattisgarh", "Madhya Pradesh"},
    "West": {"Gujarat", "Maharashtra", "Goa"},
    "South": {"Andhra Pradesh", "Karnataka", "Kerala", "Puducherry", "Tamil Nadu", "Telangana"},
    "East": {"Bihar", "Jharkhand", "Odisha", "West Bengal"},
    "Northeast": {"Arunachal Pradesh", "Assam", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Sikkim", "Tripura"},
    "Islands": {"Andaman and Nicobar Islands"},
}
STATE_TO_REGION = {state: region for region, states in STATE_REGIONS.items() for state in states}


def valid_shift(data: pd.DataFrame, grouped: pd.core.groupby.DataFrameGroupBy, column: str, lag: int) -> pd.Series:
    """Return a lag only when the source record is exactly ``lag`` calendar days old."""
    value = grouped[column].shift(lag)
    source_date = grouped["date"].shift(lag)
    return value.where((data["date"] - source_date).dt.days.eq(lag))


def build_feature_table() -> tuple[pd.DataFrame, list[str], list[str]]:
    data = pd.read_csv(DATA_PATH, parse_dates=["date"])
    data["location_key"] = data["state"].str.strip() + "||" + data["area"].str.strip()
    data["region"] = data["state"].map(STATE_TO_REGION).fillna("Other")
    data = data.sort_values(["location_key", "date"]).drop_duplicates(["location_key", "date"]).reset_index(drop=True)
    data = data.rename(columns={"aqi_value": "aqi_at_issue"})
    grouped = data.groupby("location_key", sort=False)
    derived: dict[str, pd.Series] = {}

    # Long AQI histories capture persistence, trend, and episodic build-up.
    for lag in (1, 3, 7, 14, 28):
        derived[f"aqi_lag_{lag}"] = valid_shift(data, grouped, "aqi_at_issue", lag)
    for window in (3, 7, 14, 28):
        prior = grouped["aqi_at_issue"].shift(1)
        first_date = grouped["date"].shift(window)
        is_contiguous = (data["date"] - first_date).dt.days.eq(window)
        history = prior.groupby(data["location_key"])
        for name, transform in {
            "mean": lambda values: values.rolling(window, min_periods=window).mean(),
            "max": lambda values: values.rolling(window, min_periods=window).max(),
            "std": lambda values: values.rolling(window, min_periods=window).std(),
        }.items():
            derived[f"aqi_{name}_{window}d"] = history.transform(transform).where(is_contiguous)
    derived["aqi_change_1d"] = data["aqi_at_issue"] - derived["aqi_lag_1"]
    derived["aqi_change_7d"] = data["aqi_at_issue"] - derived["aqi_lag_7"]
    derived["aqi_trend_vs_7d_mean"] = data["aqi_at_issue"] - derived["aqi_mean_7d"]
    derived["aqi_trend_vs_28d_mean"] = data["aqi_at_issue"] - derived["aqi_mean_28d"]

    # Weather is deliberately lagged: historical reanalysis values from the issue
    # day are not treated as a live operational forecast input.
    weather = [column for column in WEATHER if column in data.columns]
    for column in weather:
        for lag in (1, 3, 7, 14):
            derived[f"{column}_lag_{lag}"] = valid_shift(data, grouped, column, lag)
        prior = grouped[column].shift(1)
        for window in (7, 14):
            first_date = grouped["date"].shift(window)
            contiguous = (data["date"] - first_date).dt.days.eq(window)
            derived[f"{column}_mean_{window}d"] = prior.groupby(data["location_key"]).transform(
                lambda values: values.rolling(window, min_periods=window).mean()
            ).where(contiguous)

    for horizon in HORIZONS:
        future_date = data["date"] + pd.Timedelta(days=horizon)
        derived[f"forecast_month_h{horizon}"] = future_date.dt.month
        derived[f"forecast_doy_sin_h{horizon}"] = np.sin(2 * np.pi * future_date.dt.dayofyear / 365.25)
        derived[f"forecast_doy_cos_h{horizon}"] = np.cos(2 * np.pi * future_date.dt.dayofyear / 365.25)
        future_aqi = grouped["aqi_at_issue"].shift(-horizon)
        future_observed_date = grouped["date"].shift(-horizon)
        target = future_aqi.where((future_observed_date - data["date"]).dt.days.eq(horizon))
        derived[f"target_aqi_h{horizon}"] = target
        derived[f"event_gt_{CUTOFF}_h{horizon}"] = (target > CUTOFF).astype(float).where(target.notna())

    data = pd.concat([data, pd.DataFrame(derived, index=data.index)], axis=1)
    numeric = ["aqi_at_issue", "latitude", "longitude", *derived.keys()]
    numeric = [column for column in numeric if not column.startswith(("target_", "event_"))]
    return data, numeric, ["region", "state", "location_key"]


def make_model(numeric: list[str], categorical: list[str], positive_weight: float) -> Pipeline:
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("prep", preprocess),
        ("model", XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", n_estimators=450,
            max_depth=5, learning_rate=0.035, subsample=0.85, colsample_bytree=0.80,
            min_child_weight=5, reg_alpha=0.25, reg_lambda=2.0,
            scale_pos_weight=positive_weight, tree_method="hist", n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


def metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    issued = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, issued, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if fp + tn else np.nan
    return {
        "ROC_AUC": roc_auc_score(y_true, probability),
        "PR_AUC": average_precision_score(y_true, probability),
        "Recall": recall_score(y_true, issued, zero_division=0),
        "Precision": precision_score(y_true, issued, zero_division=0),
        "false_positive_rate": fpr,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def select_robust_threshold(validation_predictions: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Maximise mean recall while requiring FPR <= target in every validation fold."""
    rows = []
    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        fold_metrics = []
        for _, fold in validation_predictions.groupby("fold"):
            fold_metrics.append(metrics(fold.event_observed.to_numpy(), fold.probability.to_numpy(), threshold))
        table = pd.DataFrame(fold_metrics)
        rows.append({
            "threshold": threshold,
            "mean_recall": table.Recall.mean(),
            "mean_precision": table.Precision.mean(),
            "worst_fold_fpr": table.false_positive_rate.max(),
            "mean_fpr": table.false_positive_rate.mean(),
        })
    candidates = pd.DataFrame(rows)
    feasible = candidates[candidates.worst_fold_fpr <= MAX_FALSE_POSITIVE_RATE]
    if feasible.empty:
        raise RuntimeError("No threshold satisfies the FPR target in every validation fold.")
    choice = feasible.sort_values(["mean_recall", "mean_precision", "threshold"], ascending=[False, False, False]).iloc[0]
    return float(choice.threshold), candidates


def regional_probabilities(
    train: pd.DataFrame, predict: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> np.ndarray:
    """Fit independent region models; fall back to nationwide training for sparse regions."""
    output = pd.Series(index=predict.index, dtype=float)
    for region, region_predict in predict.groupby("region", observed=True):
        region_train = train[train.region.eq(region)]
        positives = int(region_train.label.sum())
        if len(region_train) < 1_000 or positives < 30:
            region_train = train
        positive_weight = (len(region_train) - region_train.label.sum()) / max(region_train.label.sum(), 1)
        model = make_model(numeric, categorical, float(positive_weight))
        model.fit(region_train[numeric + categorical], region_train.label)
        output.loc[region_predict.index] = model.predict_proba(region_predict[numeric + categorical])[:, 1]
    return output.loc[predict.index].to_numpy()


def global_probabilities(
    train: pd.DataFrame, predict: pd.DataFrame, numeric: list[str], categorical: list[str],
    return_model: bool = False,
) -> np.ndarray | tuple[np.ndarray, Pipeline]:
    positive_weight = (len(train) - train.label.sum()) / max(train.label.sum(), 1)
    model = make_model(numeric, categorical, float(positive_weight))
    model.fit(train[numeric + categorical], train.label)
    probability = model.predict_proba(predict[numeric + categorical])[:, 1]
    return (probability, model) if return_model else probability


def evaluate_variant(
    name: str, frame: pd.DataFrame, numeric: list[str], categorical: list[str], horizon: int
) -> tuple[dict, pd.DataFrame]:
    label = f"event_gt_{CUTOFF}_h{horizon}"
    usable = frame[frame[label].notna()].copy().rename(columns={label: "label"})
    usable["label"] = usable["label"].astype(int)
    # The held-out deployment test is November--April.  Use the same seasonal
    # windows from prior years so threshold selection sees winter pollution
    # regimes instead of being tuned only on the lower-pollution monsoon period.
    validation_folds = [
        ("2022-11_to_2023-04", pd.Timestamp("2022-10-31"), pd.Timestamp("2022-11-01"), pd.Timestamp("2023-04-30")),
        ("2023-11_to_2024-04", pd.Timestamp("2023-10-31"), pd.Timestamp("2023-11-01"), pd.Timestamp("2024-04-30")),
    ]
    predict_fn = regional_probabilities if name == "Regional" else global_probabilities
    validation_predictions = []
    for fold_name, train_end, valid_start, valid_end in validation_folds:
        train = usable[usable.date <= train_end]
        valid = usable[(usable.date >= valid_start) & (usable.date <= valid_end)]
        probability = predict_fn(train, valid, numeric, categorical)
        validation_predictions.append(pd.DataFrame({
            "fold": fold_name, "event_observed": valid.label.to_numpy(), "probability": probability,
        }))
    validation_predictions = pd.concat(validation_predictions, ignore_index=True)
    threshold, threshold_curve = select_robust_threshold(validation_predictions)

    train = usable[usable.date <= pd.Timestamp("2024-10-31")]
    test = usable[(usable.date >= pd.Timestamp("2024-11-01")) & (usable.date <= pd.Timestamp("2025-04-27"))].copy()
    if name == "Global":
        probability, final_model = predict_fn(train, test, numeric, categorical, return_model=True)
        artifact = {
            "model_scope": name, "horizon_days": horizon, "aqi_cutoff": CUTOFF,
            "alert_threshold": threshold, "numeric_features": numeric,
            "categorical_features": categorical,
            "trained_through": "2024-10-31", "random_state": RANDOM_STATE,
        }
        joblib.dump(
            {"pipeline": final_model, "metadata": artifact},
            MODEL_DIR / f"global_aqi_gt_{CUTOFF}_h{horizon}.joblib",
        )
        with (MODEL_DIR / f"global_aqi_gt_{CUTOFF}_h{horizon}_threshold.json").open("w", encoding="utf-8") as handle:
            json.dump(artifact, handle, indent=2)
    else:
        probability = predict_fn(train, test, numeric, categorical)
    result = {
        "model_scope": name, "horizon_days": horizon, "threshold": threshold,
        "validation_worst_fold_fpr": float(threshold_curve.loc[threshold_curve.threshold.eq(threshold), "worst_fold_fpr"].iloc[0]),
        "validation_mean_recall": float(threshold_curve.loc[threshold_curve.threshold.eq(threshold), "mean_recall"].iloc[0]),
        **metrics(test.label.to_numpy(), probability, threshold),
    }
    output = test[["date", "state", "area", "region", "location_key"]].copy()
    output["forecast_date"] = output.date + pd.Timedelta(days=horizon)
    output["model_scope"] = name
    output["horizon_days"] = horizon
    output["event_observed"] = test.label.to_numpy()
    output["probability"] = probability
    output["threshold"] = threshold
    output["alert_issued"] = (probability >= threshold).astype(int)
    threshold_curve["model_scope"] = name
    threshold_curve["horizon_days"] = horizon
    threshold_curve.to_csv(REPORT_DIR / f"threshold_curve_{name.lower()}_h{horizon}.csv", index=False)
    return result, output


def main() -> None:
    frame, numeric, categorical = build_feature_table()
    print(f"Prepared {len(frame):,} records with {len(numeric)} numeric and {len(categorical)} categorical features.")
    results, predictions = [], []
    for horizon in HORIZONS:
        for name in ("Global", "Regional"):
            print(f"Evaluating {name.lower()} AQI > {CUTOFF} warning model for t+{horizon}...")
            result, prediction = evaluate_variant(name, frame, numeric, categorical, horizon)
            results.append(result)
            predictions.append(prediction)
    results_df = pd.DataFrame(results).sort_values(["horizon_days", "model_scope"])
    predictions_df = pd.concat(predictions, ignore_index=True)
    results_df.to_csv(REPORT_DIR / "regional_vs_global_test_metrics.csv", index=False)
    predictions_df.to_csv(REPORT_DIR / "regional_vs_global_test_predictions.csv", index=False)
    with (REPORT_DIR / "regional_early_warning_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "input": str(DATA_PATH.relative_to(ROOT)), "cutoff": CUTOFF,
            "horizons": HORIZONS, "validation_folds": ["2022-11_to_2023-04", "2023-11_to_2024-04"],
            "final_train_end": "2024-10-31", "final_test_period": "2024-11-01 to 2025-04-27",
            "max_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
        }, handle, indent=2)
    print(results_df.round(3).to_string(index=False))
    print(f"Saved results to {REPORT_DIR}")


if __name__ == "__main__":
    main()
