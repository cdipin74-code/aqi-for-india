"""Joint calibration of the two-head warning system.

Each head was tuned to a 5% false-positive rate within its *own* origin
population, so their union overshoots the budget.  Worse, the naive continuation
rule ("the area is already above 200, keep warning") cannot meet a 5% budget at
all: on the validation seasons it produces a 6.2-6.7% false-positive rate by
itself.  Any honest system comparison therefore has to calibrate the continuation
trigger as well.

The two heads act on disjoint populations (origins already above the limit versus
clean origins), so their true and false positives add.  That makes the joint
search cheap: precompute counts for every continuation level and every onset
threshold, then scan the outer sum.

Selection rule is unchanged from the rest of the study: maximise mean validation
recall subject to system FPR <= 5% in *every* validation season.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from train_regional_early_warning import (
    CUTOFF,
    HORIZONS,
    MAX_FALSE_POSITIVE_RATE,
    ROOT,
    build_feature_table,
)

REPORT_DIR = ROOT / "reports" / "onset"
ONSET_VALIDATION = REPORT_DIR / "onset_validation_predictions.csv"
ONSET_TEST = REPORT_DIR / "onset_warning_test_predictions.csv"

VALIDATION_FOLDS = [
    ("2022-11_to_2023-04", pd.Timestamp("2022-11-01"), pd.Timestamp("2023-04-30")),
    ("2023-11_to_2024-04", pd.Timestamp("2023-11-01"), pd.Timestamp("2024-04-30")),
]
TEST_START = pd.Timestamp("2024-11-01")
TEST_END = pd.Timestamp("2025-04-27")

CONTINUATION_LEVELS = np.arange(CUTOFF, 401, 5)
ONSET_THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def assemble(frame: pd.DataFrame, horizon: int, predictions: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    label = f"event_gt_{CUTOFF}_h{horizon}"
    rows = frame[frame[label].notna()][["date", "location_key", "aqi_at_issue", label]].copy()
    rows["event_observed"] = rows[label].astype(int)
    if start is not None:
        rows = rows[(rows.date >= start) & (rows.date <= end)]
    predictions = predictions[predictions.horizon_days.eq(horizon)][["date", "location_key", "probability"]]
    rows = rows.merge(predictions.rename(columns={"probability": "onset_probability"}),
                      on=["date", "location_key"], how="left")
    rows["already_bad"] = rows["aqi_at_issue"].gt(CUTOFF)
    return rows


def counts(observed: np.ndarray, issued: np.ndarray) -> tuple[int, int, int, int]:
    tn = int(((~issued) & (observed == 0)).sum())
    fp = int((issued & (observed == 0)).sum())
    fn = int(((~issued) & (observed == 1)).sum())
    tp = int((issued & (observed == 1)).sum())
    return tn, fp, fn, tp


def summarise(tn: int, fp: int, fn: int, tp: int) -> dict:
    recall = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn, "Recall": recall, "Precision": precision,
        "F1": 2 * precision * recall / (precision + recall) if precision and recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
        "alerts_issued": int(tp + fp),
    }


def main() -> None:
    frame, _, _ = build_feature_table()
    validation = pd.read_csv(ONSET_VALIDATION, parse_dates=["date"])
    test_predictions = pd.read_csv(ONSET_TEST, parse_dates=["date"])

    results, curve_rows = [], []

    for horizon in HORIZONS:
        recall_by_fold, fpr_by_fold = [], []
        for fold_name, start, end in VALIDATION_FOLDS:
            fold = assemble(frame, horizon, validation, start, end)
            observed = fold.event_observed.to_numpy().astype(int)
            level = fold.aqi_at_issue.to_numpy()
            onset = fold.onset_probability.fillna(0.0).to_numpy()
            clean = ~fold.already_bad.to_numpy()
            n_pos = int((observed == 1).sum())
            n_neg = int((observed == 0).sum())

            cont_tp = np.array([((level >= c) & (observed == 1)).sum() for c in CONTINUATION_LEVELS])
            cont_fp = np.array([((level >= c) & (observed == 0)).sum() for c in CONTINUATION_LEVELS])
            onset_tp = np.array([((onset >= t) & clean & (observed == 1)).sum() for t in ONSET_THRESHOLDS])
            onset_fp = np.array([((onset >= t) & clean & (observed == 0)).sum() for t in ONSET_THRESHOLDS])

            tp = cont_tp[:, None] + onset_tp[None, :]
            fp = cont_fp[:, None] + onset_fp[None, :]
            recall_by_fold.append(tp / n_pos)
            fpr_by_fold.append(fp / n_neg)

        mean_recall = np.mean(recall_by_fold, axis=0)
        worst_fpr = np.max(fpr_by_fold, axis=0)
        feasible = worst_fpr <= MAX_FALSE_POSITIVE_RATE
        if not feasible.any():
            raise RuntimeError(f"No jointly feasible configuration for t+{horizon}.")

        masked = np.where(feasible, mean_recall, -np.inf)
        best = np.unravel_index(np.argmax(masked), masked.shape)
        level_choice = float(CONTINUATION_LEVELS[best[0]])
        threshold_choice = float(ONSET_THRESHOLDS[best[1]])

        for i, level in enumerate(CONTINUATION_LEVELS):
            for j, threshold in enumerate(ONSET_THRESHOLDS):
                curve_rows.append({
                    "horizon_days": horizon,
                    "continuation_level": float(level),
                    "onset_threshold": float(threshold),
                    "mean_validation_recall": float(mean_recall[i, j]),
                    "worst_fold_fpr": float(worst_fpr[i, j]),
                    "feasible": bool(feasible[i, j]),
                })

        test = assemble(frame, horizon, test_predictions)
        test = test[(test.date >= TEST_START) & (test.date <= TEST_END)]
        observed = test.event_observed.to_numpy().astype(int)
        level = test.aqi_at_issue.to_numpy()
        onset = test.onset_probability.fillna(0.0).to_numpy()
        already_bad = test.already_bad.to_numpy()

        configurations = {
            "two-head, jointly calibrated": (level >= level_choice) | (onset >= threshold_choice),
            "persistence rule (alert iff already above limit)": already_bad,
            "onset head only": onset >= threshold_choice,
        }
        for name, issued in configurations.items():
            results.append({
                "horizon_days": horizon,
                "system": name,
                "continuation_level": level_choice if name != "persistence rule (alert iff already above limit)" else CUTOFF,
                "onset_threshold": threshold_choice,
                **summarise(*counts(observed, issued)),
            })

    table = pd.DataFrame(results).sort_values(["horizon_days", "system"])
    table.to_csv(REPORT_DIR / "combined_system_calibrated.csv", index=False)
    curve = pd.DataFrame(curve_rows)
    curve[curve.horizon_days.eq(1)].to_csv(REPORT_DIR / "combined_system_threshold_curve_h1.csv", index=False)

    with (REPORT_DIR / "combined_system_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "continuation_rule": "alert when AQI at issue time is at or above the calibrated level",
            "onset_rule": "alert when the clean-origin onset model exceeds the calibrated threshold",
            "calibration": "joint grid; maximise mean validation recall subject to system FPR <= target in every fold",
            "continuation_levels_tested": [float(CONTINUATION_LEVELS.min()), float(CONTINUATION_LEVELS.max())],
            "max_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
            "final_test_period": f"{TEST_START.date()} to {TEST_END.date()}",
            "note": "Heads act on disjoint origin populations, so counts add exactly.",
        }, handle, indent=2)

    print(table.round(3).to_string(index=False))
    print(f"\nSaved to {REPORT_DIR}")


if __name__ == "__main__":
    main()
