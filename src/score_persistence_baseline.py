"""Threshold-free persistence baseline for the AQI >200 warning task.

The rule-based persistence baseline in ``run_feature_ablation.py`` sits at a fixed
operating point (alert iff AQI_t > cutoff), which makes it hard to compare with a
model that is tuned to a 5% false-positive budget.  Here the *same* probabilistic
treatment is applied to persistence: AQI at issue time is used directly as a
ranking score, so PR-AUC and the threshold-selection rule are directly
comparable with the machine-learning models.
"""

from __future__ import annotations

import json

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
    metrics,
    select_robust_threshold,
)

REPORT_DIR = ROOT / "reports" / "ablation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_FOLDS = [
    ("2022-11_to_2023-04", pd.Timestamp("2022-10-31"), pd.Timestamp("2022-11-01"), pd.Timestamp("2023-04-30")),
    ("2023-11_to_2024-04", pd.Timestamp("2023-10-31"), pd.Timestamp("2023-11-01"), pd.Timestamp("2024-04-30")),
]
FINAL_TRAIN_END = pd.Timestamp("2024-10-31")
TEST_START = pd.Timestamp("2024-11-01")
TEST_END = pd.Timestamp("2025-04-27")


def normalise_aqi_scores(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """Map current AQI onto [0, 1] using reference-period min/max.

    A monotone transform leaves PR-AUC and ROC-AUC unchanged; it only puts the
    persistence score on the same 0-1 scale as a model probability so the same
    threshold-selection routine can be reused.
    """
    low, high = reference["aqi_at_issue"].min(), reference["aqi_at_issue"].max()
    return ((frame["aqi_at_issue"] - low) / (high - low)).clip(0, 1)


def main() -> None:
    frame, _, _ = build_feature_table()
    rows = []
    for horizon in HORIZONS:
        label = f"event_gt_{CUTOFF}_h{horizon}"
        usable = frame[frame[label].notna()].copy().rename(columns={label: "label"})
        usable["label"] = usable["label"].astype(int)

        train = usable[usable.date <= FINAL_TRAIN_END]
        test = usable[(usable.date >= TEST_START) & (usable.date <= TEST_END)].copy()

        # Threshold selection uses the identical rule as the ML models.
        validation_predictions = []
        for fold_name, train_end, valid_start, valid_end in VALIDATION_FOLDS:
            fold_train = usable[usable.date <= train_end]
            fold_valid = usable[(usable.date >= valid_start) & (usable.date <= valid_end)].copy()
            fold_valid["score"] = normalise_aqi_scores(fold_valid, fold_train)
            validation_predictions.append(pd.DataFrame({
                "fold": fold_name,
                "event_observed": fold_valid.label.to_numpy(),
                "probability": fold_valid.score.to_numpy(),
            }))
        validation_predictions = pd.concat(validation_predictions, ignore_index=True)
        try:
            threshold, _ = select_robust_threshold(validation_predictions)
            feasible = True
        except RuntimeError:
            threshold, feasible = np.nan, False

        test["score"] = normalise_aqi_scores(test, train)
        result = {
            "horizon_days": horizon,
            "baseline": "persistence score (AQI at issue time, min-max scaled)",
            "ROC_AUC": roc_auc_score(test.label, test.score),
            "PR_AUC": average_precision_score(test.label, test.score),
            "test_prevalence": float(test.label.mean()),
            "threshold": threshold,
            "fpr_constraint_feasible": feasible,
        }
        if feasible:
            result.update(metrics(test.label.to_numpy(), test.score.to_numpy(), threshold))
        rows.append(result)

    output = pd.DataFrame(rows)
    output.to_csv(REPORT_DIR / "persistence_score_baseline.csv", index=False)
    with (REPORT_DIR / "persistence_score_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "input": str(DATA_PATH.relative_to(ROOT)),
            "cutoff": CUTOFF,
            "score": "aqi_at_issue scaled by training-period min/max",
            "validation_folds": [fold[0] for fold in VALIDATION_FOLDS],
            "final_test_period": f"{TEST_START.date()} to {TEST_END.date()}",
            "max_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
            "note": "Identical threshold rule as the ML models, applied to the persistence score.",
        }, handle, indent=2)
    print(output.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
