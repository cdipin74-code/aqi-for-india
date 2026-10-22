"""Combined warning system: continuation head plus onset head.

A deployed system faces two different situations, and they need different logic:

* the area is **already** above the limit - the useful question is whether it
  stays there, which is dominated by persistence and handled by the main model;
* the area is **currently clean** - persistence is structurally blind, so the
  onset model is the only component that can produce a warning.

The two heads are trained on disjoint origin populations and combined here into
one alert stream, then compared at system level against the persistence rule
("alert iff AQI(t) already exceeds the limit").  This makes the incremental value
of the onset head visible on the complete set of hazardous days rather than only
inside the clean-origin subset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from train_regional_early_warning import CUTOFF, DATA_PATH, HORIZONS, ROOT

REPORT_DIR = ROOT / "reports" / "onset"
ONSET_METRICS = REPORT_DIR / "onset_warning_metrics.csv"
ONSET_PREDICTIONS = REPORT_DIR / "onset_warning_test_predictions.csv"
CONTINUATION_PREDICTIONS = ROOT / "reports" / "regional_early_warning" / "regional_vs_global_test_predictions.csv"


def load_issue_aqi() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, usecols=["date", "state", "area", "aqi_value"], parse_dates=["date"])
    data["location_key"] = data["state"].str.strip() + "||" + data["area"].str.strip()
    return data[["date", "location_key", "aqi_value"]].drop_duplicates(["date", "location_key"])


def system_metrics(observed: np.ndarray, issued: np.ndarray) -> dict:
    tn = int(((~issued) & (observed == 0)).sum())
    fp = int((issued & (observed == 0)).sum())
    fn = int(((~issued) & (observed == 1)).sum())
    tp = int((issued & (observed == 1)).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Recall": recall, "Precision": precision,
        "F1": 2 * precision * recall / (precision + recall) if precision and recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
        "alerts_issued": int(issued.sum()),
    }


def main() -> None:
    onset_metrics = pd.read_csv(ONSET_METRICS)
    onset_predictions = pd.read_csv(ONSET_PREDICTIONS, parse_dates=["date"])
    continuation = pd.read_csv(CONTINUATION_PREDICTIONS, parse_dates=["date"])
    continuation = continuation[continuation.model_scope.eq("Global")]

    rows = []
    for horizon in HORIZONS:
        onset_threshold = float(
            onset_metrics.loc[
                onset_metrics.horizon_days.eq(horizon) & onset_metrics.score.eq("XGBoost onset model"),
                "threshold",
            ].iloc[0]
        )
        base = continuation[continuation.horizon_days.eq(horizon)][
            ["date", "location_key", "event_observed", "probability", "threshold"]
        ].rename(columns={"probability": "continuation_probability", "threshold": "continuation_threshold"})
        onset = onset_predictions[onset_predictions.horizon_days.eq(horizon)][
            ["date", "location_key", "probability"]
        ].rename(columns={"probability": "onset_probability"})

        frame = base.merge(onset, on=["date", "location_key"], how="left")
        frame = frame.merge(load_issue_aqi(), on=["date", "location_key"], how="left", validate="one_to_one")

        observed = frame.event_observed.to_numpy().astype(int)
        already_bad = frame.aqi_value.gt(CUTOFF).to_numpy()
        continuation_alert = frame.continuation_probability.to_numpy() >= frame.continuation_threshold.to_numpy()
        onset_alert = frame.onset_probability.fillna(0.0).to_numpy() >= onset_threshold

        if not np.array_equal(already_bad, frame.onset_probability.isna().to_numpy()):
            raise RuntimeError("Onset predictions and the clean-origin mask disagree.")

        combined = np.where(already_bad, continuation_alert, onset_alert)

        rows.append({
            "horizon_days": horizon, "system": "persistence only (alert iff already above limit)",
            **system_metrics(observed, already_bad),
        })
        rows.append({
            "horizon_days": horizon, "system": "continuation model only",
            **system_metrics(observed, continuation_alert),
        })
        rows.append({
            "horizon_days": horizon, "system": "onset head only",
            **system_metrics(observed, onset_alert),
        })
        rows.append({
            "horizon_days": horizon, "system": "two-head (continuation + onset)",
            **system_metrics(observed, combined),
        })

    table = pd.DataFrame(rows).sort_values(["horizon_days", "Recall"], ascending=[True, False])
    table.to_csv(REPORT_DIR / "combined_system_metrics.csv", index=False)
    print(table.round(3).to_string(index=False))
    print(f"\nSaved to {REPORT_DIR / 'combined_system_metrics.csv'}")


if __name__ == "__main__":
    main()
