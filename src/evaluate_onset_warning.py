"""Episode-onset warning: the task where persistence is structurally useless.

The standard "will AQI exceed 200 tomorrow" target is dominated by persistence,
because most hazardous days are followed by hazardous days.  A warning system is
only valuable at the moment conditions *transition*: today is acceptable and the
next one to three days become hazardous.

This script defines onset as

    onset(h) = 1[ AQI(t) <= 200 and AQI(t+h) > 200 ]

and evaluates only origins with AQI(t) <= 200, since onset is impossible
elsewhere.  On this population a persistence rule predicts "no alert" for every
row and therefore has zero recall by construction.  The meaningful competitors
are therefore:

* the no-skill prevalence (PR-AUC of a random score);
* an area-month climatology score estimated only from the training period;
* an AQI-level score, i.e. "warn when today is already close to the limit",
  which is the best rule that ignores everything except the current reading;
* the gradient-boosted model on leakage-safe history, weather, and neighbouring
  area features.

All four are ranked scores evaluated with the identical threshold rule
(maximum mean validation recall subject to FPR <= 5% in every fold) so the
comparison is at a matched false-alarm budget rather than at hand-picked
operating points.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from train_regional_early_warning import (
    CUTOFF,
    DATA_PATH,
    HORIZONS,
    MAX_FALSE_POSITIVE_RATE,
    ROOT,
    build_feature_table,
    global_probabilities,
    metrics,
    select_robust_threshold,
)

REPORT_DIR = ROOT / "reports" / "onset"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = ROOT / "models" / "onset"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_FOLDS = [
    ("2022-11_to_2023-04", pd.Timestamp("2022-10-31"), pd.Timestamp("2022-11-01"), pd.Timestamp("2023-04-30")),
    ("2023-11_to_2024-04", pd.Timestamp("2023-10-31"), pd.Timestamp("2023-11-01"), pd.Timestamp("2024-04-30")),
]
FINAL_TRAIN_END = pd.Timestamp("2024-10-31")
TEST_START = pd.Timestamp("2024-11-01")
TEST_END = pd.Timestamp("2025-04-27")
NEIGHBOURS = 5


def haversine_matrix(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    lat = np.radians(latitude)
    lon = np.radians(longitude)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def add_neighbour_features(frame: pd.DataFrame, k: int = NEIGHBOURS) -> tuple[pd.DataFrame, list[str]]:
    """Mean/max AQI and mean 1-day change across each area's ``k`` nearest areas.

    Pollution transport is spatially organised, and neighbouring areas report on
    the same day, so these values are available at issue time.  Wind direction is
    not present in the dataset, so this is adjacency rather than true upwind
    advection; that limitation is stated in the manuscript.
    """
    locations = frame.groupby("location_key")[["latitude", "longitude"]].first()
    keys = locations.index.to_numpy()
    distance = haversine_matrix(locations.latitude.to_numpy(), locations.longitude.to_numpy())
    np.fill_diagonal(distance, np.inf)
    nearest = np.argsort(distance, axis=1)[:, :k]

    weights = np.zeros_like(distance)
    weights[np.arange(len(keys))[:, None], nearest] = 1.0 / k

    pivot = frame.pivot_table(index="date", columns="location_key", values="aqi_at_issue", aggfunc="mean")
    pivot = pivot.reindex(columns=keys)
    values = pivot.to_numpy(dtype=float)
    observed = np.isfinite(values).astype(float)
    filled = np.nan_to_num(values, nan=0.0)

    numerator = filled @ weights.T
    denominator = observed @ weights.T
    with np.errstate(invalid="ignore", divide="ignore"):
        neighbour_mean = np.where(denominator > 0, numerator / denominator, np.nan)

    stacked = values[:, nearest]                      # (dates, locations, k)
    with np.errstate(invalid="ignore"):
        neighbour_max = np.nanmax(np.where(np.isfinite(stacked), stacked, -np.inf), axis=2)
    neighbour_max[~np.isfinite(neighbour_max)] = np.nan

    def to_long(matrix: np.ndarray, name: str) -> pd.DataFrame:
        wide = pd.DataFrame(matrix, index=pivot.index, columns=pivot.columns)
        # Build the long form with numpy so missing values are preserved across
        # pandas versions (``stack(dropna=False)`` is deprecated).
        return pd.DataFrame({
            "date": np.repeat(wide.index.to_numpy(), wide.shape[1]),
            "location_key": np.tile(wide.columns.to_numpy(), wide.shape[0]),
            name: wide.to_numpy().ravel(),
        })

    frame = frame.merge(to_long(neighbour_mean, "neighbour_mean_aqi"), on=["date", "location_key"], how="left")
    frame = frame.merge(to_long(neighbour_max, "neighbour_max_aqi"), on=["date", "location_key"], how="left")
    frame = frame.merge(to_long(denominator, "neighbour_count_reporting"), on=["date", "location_key"], how="left")
    frame["neighbour_mean_aqi_change_1d"] = frame.groupby("location_key")["neighbour_mean_aqi"].diff()

    return frame, ["neighbour_mean_aqi", "neighbour_max_aqi", "neighbour_mean_aqi_change_1d"]


def normalise(score: pd.Series, reference: pd.Series) -> np.ndarray:
    """Map a raw score onto [0, 1] using training-period extremes.

    Missing values (for example a trend for an area's first observed day) are
    treated as "no evidence of the condition", which keeps the ranking score
    finite and is the conservative choice for an alerting rule.
    """
    low, high = float(reference.min()), float(reference.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(len(score))
    scaled = ((score - low) / (high - low)).clip(0, 1).to_numpy(dtype=float)
    return np.nan_to_num(scaled, nan=0.0)


def climatology_score(train: pd.DataFrame, target: pd.DataFrame, horizon: int) -> np.ndarray:
    """Area-month onset rate learned only on training data."""
    train = train.assign(forecast_month=(train["date"] + pd.Timedelta(days=horizon)).dt.month)
    target = target.assign(forecast_month=(target["date"] + pd.Timedelta(days=horizon)).dt.month)
    rates = train.groupby(["location_key", "forecast_month"]).label.mean().rename("rate").reset_index()
    merged = target[["location_key", "forecast_month"]].merge(rates, on=["location_key", "forecast_month"], how="left")
    return merged.rate.fillna(float(train.label.mean())).to_numpy()


def evaluate_score(
    score: np.ndarray, y: np.ndarray, validation_scores: list[tuple[np.ndarray, np.ndarray]]
) -> dict:
    validation = pd.concat([
        pd.DataFrame({"fold": fold, "event_observed": labels, "probability": values})
        for fold, (values, labels) in zip([f[0] for f in VALIDATION_FOLDS], validation_scores)
    ], ignore_index=True)
    try:
        threshold, _ = select_robust_threshold(validation)
        feasible = True
    except RuntimeError:
        threshold, feasible = np.nan, False

    result = {
        "ROC_AUC": roc_auc_score(y, score),
        "PR_AUC": average_precision_score(y, score),
        "threshold": threshold,
        "fpr_constraint_feasible": feasible,
    }
    if feasible:
        result.update(metrics(y, score, threshold))
    return result


def main() -> None:
    frame, numeric, _ = build_feature_table()
    if "aqi_lag_3" in frame.columns:
        frame["aqi_change_3d"] = frame["aqi_at_issue"] - frame["aqi_lag_3"]
    frame, neighbour_features = add_neighbour_features(frame)
    numeric = numeric + neighbour_features
    categorical = ["region", "state", "location_key"]
    print(f"Features: {len(numeric)} numeric ({len(neighbour_features)} neighbour) + {len(categorical)} categorical")

    rows, prediction_frames = [], []
    validation_output: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        future = f"target_aqi_h{horizon}"
        population = frame[frame[future].notna() & frame["aqi_at_issue"].le(CUTOFF)].copy()
        population["label"] = population[future].gt(CUTOFF).astype(int)
        print(
            f"t+{horizon}: {len(population):,} clean origins, "
            f"{int(population.label.sum()):,} onsets ({population.label.mean():.3%})"
        )

        train = population[population.date <= FINAL_TRAIN_END]
        test = population[(population.date >= TEST_START) & (population.date <= TEST_END)].copy()
        y_test = test.label.to_numpy()

        validation_scores = []
        validation_frames = []
        for fold_name, train_end, valid_start, valid_end in VALIDATION_FOLDS:
            fold_train = population[population.date <= train_end]
            fold_valid = population[(population.date >= valid_start) & (population.date <= valid_end)]
            probability = global_probabilities(fold_train, fold_valid, numeric, categorical)
            validation_scores.append((probability, fold_valid.label.to_numpy()))
            validation_frames.append(pd.DataFrame({
                "fold": fold_name,
                "date": fold_valid.date.to_numpy(),
                "location_key": fold_valid.location_key.to_numpy(),
                "event_observed": fold_valid.label.to_numpy(),
                "probability": probability,
            }))

        model_probability, final_model = global_probabilities(
            train, test, numeric, categorical, return_model=True
        )

        # Rule baselines.  A "trend" rule is the classic onset heuristic ("AQI has
        # been climbing, so warn") and must be beaten by the learned model, not
        # merely by a threshold on today's reading.
        baseline_scorers = {
            "climatology (area-month onset rate)": lambda fit, eval_: climatology_score(fit, eval_, horizon),
            "AQI level (warn when today is near the limit)": (
                lambda fit, eval_: normalise(eval_["aqi_at_issue"], fit["aqi_at_issue"])
            ),
            "AQI 1-day trend (warn when AQI is rising)": (
                lambda fit, eval_: normalise(eval_["aqi_change_1d"], fit["aqi_change_1d"])
            ),
            "AQI 3-day trend (warn when the 3-day trend is rising)": (
                lambda fit, eval_: normalise(eval_["aqi_change_3d"], fit["aqi_change_3d"])
            ),
            "AQI 7-day trend (warn when the week is worsening)": (
                lambda fit, eval_: normalise(eval_["aqi_change_7d"], fit["aqi_change_7d"])
            ),
        }

        model_result = evaluate_score(model_probability, y_test, validation_scores)
        rows.append({
            "horizon_days": horizon,
            "score": "XGBoost onset model",
            "test_origins": len(test),
            "test_onsets": int(y_test.sum()),
            "test_onset_rate": float(y_test.mean()),
            **model_result,
        })

        joblib.dump(final_model, MODEL_DIR / f"onset_aqi_gt_{CUTOFF}_h{horizon}.joblib")
        with (MODEL_DIR / f"onset_aqi_gt_{CUTOFF}_h{horizon}_threshold.json").open("w", encoding="utf-8") as handle:
            json.dump({
                "task": f"onset: AQI(t) <= {CUTOFF} and AQI(t+{horizon}) > {CUTOFF}",
                "horizon_days": horizon,
                "alert_threshold": model_result["threshold"],
                "numeric_features": numeric,
                "categorical_features": categorical,
                "training_rows": int(len(train)),
                "training_positives": int(train.label.sum()),
                "training_end": str(FINAL_TRAIN_END.date()),
                "validation_folds": [fold[0] for fold in VALIDATION_FOLDS],
                "test_metrics": {key: value for key, value in model_result.items() if key != "threshold"},
            }, handle, indent=2)

        for name, scorer in baseline_scorers.items():
            fold_scores = []
            for _, train_end, valid_start, valid_end in VALIDATION_FOLDS:
                fold_train = population[population.date <= train_end]
                fold_valid = population[(population.date >= valid_start) & (population.date <= valid_end)]
                fold_scores.append((scorer(fold_train, fold_valid), fold_valid.label.to_numpy()))

            result = evaluate_score(np.asarray(scorer(train, test)), y_test, fold_scores)
            rows.append({
                "horizon_days": horizon,
                "score": name,
                "test_origins": len(test),
                "test_onsets": int(y_test.sum()),
                "test_onset_rate": float(y_test.mean()),
                **result,
            })

        # Persistence is degenerate on this population: it never alerts.
        rows.append({
            "horizon_days": horizon, "score": "persistence (today is clean -> no alert)",
            "test_origins": len(test), "test_onsets": int(y_test.sum()),
            "test_onset_rate": float(y_test.mean()),
            "ROC_AUC": np.nan, "PR_AUC": float(y_test.mean()),
            "threshold": np.nan, "fpr_constraint_feasible": np.nan,
            "Recall": 0.0, "Precision": np.nan, "false_positive_rate": 0.0,
            "TP": 0, "FP": 0, "FN": int(y_test.sum()), "TN": int((1 - y_test).sum()),
        })

        output = test[["date", "state", "area", "region", "location_key"]].copy()
        output["forecast_date"] = output.date + pd.Timedelta(days=horizon)
        output["horizon_days"] = horizon
        output["event_observed"] = y_test
        output["probability"] = model_probability
        prediction_frames.append(output)

        validation_output.append(pd.concat(validation_frames, ignore_index=True).assign(horizon_days=horizon))

    table = pd.DataFrame(rows).sort_values(["horizon_days", "PR_AUC"], ascending=[True, False])
    table.to_csv(REPORT_DIR / "onset_warning_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        REPORT_DIR / "onset_warning_test_predictions.csv", index=False
    )
    pd.concat(validation_output, ignore_index=True).to_csv(
        REPORT_DIR / "onset_validation_predictions.csv", index=False
    )

    with (REPORT_DIR / "onset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "input": str(DATA_PATH.relative_to(ROOT)),
            "cutoff": CUTOFF,
            "definition": "onset(h) = AQI(t) <= cutoff AND AQI(t+h) > cutoff",
            "population": "origins with AQI(t) <= cutoff only",
            "horizons": list(HORIZONS),
            "validation_folds": [fold[0] for fold in VALIDATION_FOLDS],
            "final_train_end": str(FINAL_TRAIN_END.date()),
            "final_test_period": f"{TEST_START.date()} to {TEST_END.date()}",
            "max_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
            "neighbour_count": NEIGHBOURS,
            "neighbour_caveat": "adjacency only; wind direction unavailable, so not true upwind transport",
            "baselines": [
                "prevalence (no skill)",
                "area-month climatology",
                "AQI level",
                "1-day, 3-day and 7-day AQI trend",
                "persistence (zero recall by construction)",
            ],
            "saved_models": f"models/onset/onset_aqi_gt_{CUTOFF}_h{{1,2,3}}.joblib",
        }, handle, indent=2)

    print("\nOnset warning on the held-out November 2024 - April 2025 period:")
    print(table.drop(columns=["test_origins", "test_onsets"]).round(3).to_string(index=False))
    print(f"\nSaved to {REPORT_DIR}")


if __name__ == "__main__":
    main()
