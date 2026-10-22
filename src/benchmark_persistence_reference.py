"""Persistence reference row for the eight-model metro benchmark.

The Table 1 benchmark in the manuscript compares eight learned regressors but has
no trivial baseline, which makes it impossible to judge how much of the reported
R² is attributable to the learning algorithm rather than to day-to-day AQI
autocorrelation.  This script evaluates ``AQI(t+h) = AQI(t)`` on exactly the same
rows that the benchmark used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_regional_early_warning import ROOT

PREDICTIONS = ROOT / "reports" / "research_ml" / "benchmark_t_plus_1_predictions.csv"
METRO = ROOT / "data" / "processed" / "aqi_prediction_metro.csv"
REPORT_DIR = ROOT / "reports" / "research_ml"


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    metro = pd.read_csv(METRO, parse_dates=["date"], usecols=["date", "area", "aqi_value"])

    # One row per forecasting origin, independent of which model produced it.
    origins = predictions.drop_duplicates(["date", "area"])[["date", "area", "target_aqi_h1"]]
    merged = origins.merge(metro, on=["date", "area"], how="left", validate="one_to_one")
    missing = int(merged.aqi_value.isna().sum())
    if missing:
        raise RuntimeError(f"{missing} benchmark rows have no current-day AQI for the persistence baseline.")

    observed = merged.target_aqi_h1.to_numpy()
    predicted = merged.aqi_value.to_numpy()
    row = {
        "horizon_days": 1,
        "model": "Persistence (AQI at t)",
        "n_rows": len(merged),
        "MAE": mean_absolute_error(observed, predicted),
        "RMSE": float(np.sqrt(mean_squared_error(observed, predicted))),
        "R2": r2_score(observed, predicted),
        "within_10_pct": float((np.abs(observed - predicted) <= 10).mean() * 100),
    }

    table = pd.DataFrame([row])
    table.to_csv(REPORT_DIR / "benchmark_persistence_reference.csv", index=False)

    print(f"Evaluated on {row['n_rows']:,} rows shared with the eight-model benchmark.")
    print(table.round(3).to_string(index=False))

    model_scores = predictions.groupby("model").apply(
        lambda frame: pd.Series({
            "MAE": mean_absolute_error(frame.target_aqi_h1, frame.predicted_aqi),
        }),
        include_groups=False,
    )
    better = model_scores[model_scores.MAE < row["MAE"]]
    print(f"\nLearned models beating persistence on MAE: {len(better)} of {len(model_scores)}")
    print(better.round(2).to_string())


if __name__ == "__main__":
    main()
