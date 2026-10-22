"""Diagnostic: how much headroom would *perfect* future weather add?

This is not a deployable model.  It is an upper-bound experiment that answers a
single question: is the AQI >200 warning ceiling set by the model, or by the fact
that we do not know tomorrow's weather?

Two variants are trained on identical rows, folds, and threshold rule:

* ``current inputs``  - everything available at issue time t;
* ``oracle weather``  - the same, plus reanalysis weather *on the target date*.

Because the oracle variant uses weather from t+h, it cannot be deployed and its
score must never be reported as system performance.  Its only purpose is to
measure how much of the remaining error is meteorological.  A small gain means
operational NWP forecasts would not rescue the AQI >200 task; a large gain
justifies replacing reanalysis with forecast products.
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
)

REPORT_DIR = ROOT / "reports" / "headroom"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

VALIDATION_FOLDS = [
    ("2022-11_to_2023-04", pd.Timestamp("2022-10-31"), pd.Timestamp("2022-11-01"), pd.Timestamp("2023-04-30")),
    ("2023-11_to_2024-04", pd.Timestamp("2023-10-31"), pd.Timestamp("2023-11-01"), pd.Timestamp("2024-04-30")),
]
FINAL_TRAIN_END = pd.Timestamp("2024-10-31")
TEST_START = pd.Timestamp("2024-11-01")
TEST_END = pd.Timestamp("2025-04-27")


def attach_future_weather(frame: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    """Join target-date reanalysis weather onto each origin. Oracle only."""
    columns = [column for column in WEATHER if column in frame.columns]
    lookup = frame[["location_key", "date", *columns]].copy()
    lookup["date"] = lookup["date"] - pd.Timedelta(days=horizon)
    renamed = {column: f"oracle_{column}_at_target" for column in columns}
    lookup = lookup.rename(columns=renamed)
    merged = frame.merge(lookup, on=["location_key", "date"], how="left")
    return merged, list(renamed.values())


def main() -> None:
    frame, numeric, categorical = build_feature_table()
    rows = []

    for horizon in HORIZONS:
        label = f"event_gt_{CUTOFF}_h{horizon}"
        usable = frame[frame[label].notna()].copy().rename(columns={label: "label"})
        usable["label"] = usable["label"].astype(int)
        usable, oracle_columns = attach_future_weather(usable, horizon)

        train = usable[usable.date <= FINAL_TRAIN_END]
        test = usable[(usable.date >= TEST_START) & (usable.date <= TEST_END)].copy()
        y_test = test.label.to_numpy()

        for variant, columns in (
            ("current inputs", numeric),
            ("oracle weather (not deployable)", numeric + oracle_columns),
        ):
            print(f"t+{horizon}: {variant} ({len(columns)} features)")
            validation = []
            for fold_name, train_end, valid_start, valid_end in VALIDATION_FOLDS:
                fold_train = usable[usable.date <= train_end]
                fold_valid = usable[(usable.date >= valid_start) & (usable.date <= valid_end)]
                probability = global_probabilities(fold_train, fold_valid, columns, categorical)
                validation.append(pd.DataFrame({
                    "fold": fold_name, "event_observed": fold_valid.label.to_numpy(), "probability": probability,
                }))
            validation = pd.concat(validation, ignore_index=True)
            try:
                threshold, _ = select_robust_threshold(validation)
                feasible = True
            except RuntimeError:
                threshold, feasible = np.nan, False

            probability = global_probabilities(train, test, columns, categorical)
            result = {
                "horizon_days": horizon,
                "variant": variant,
                "n_features": len(columns),
                "threshold": threshold,
                "fpr_constraint_feasible": feasible,
            }
            if feasible:
                result.update(metrics(y_test, probability, threshold))
            else:
                from sklearn.metrics import average_precision_score, roc_auc_score

                result.update({
                    "ROC_AUC": roc_auc_score(y_test, probability),
                    "PR_AUC": average_precision_score(y_test, probability),
                })
            rows.append(result)

    table = pd.DataFrame(rows)
    table["PR_AUC"] = table["PR_AUC"].astype(float)
    table.to_csv(REPORT_DIR / "weather_headroom_diagnostic.csv", index=False)

    wide = table.pivot(index="horizon_days", columns="variant", values="PR_AUC")
    wide["oracle_gain_PR_AUC"] = wide["oracle weather (not deployable)"] - wide["current inputs"]
    wide.to_csv(REPORT_DIR / "weather_headroom_summary.csv")

    with (REPORT_DIR / "headroom_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "input": str(DATA_PATH.relative_to(ROOT)),
            "cutoff": CUTOFF,
            "warning": (
                "The oracle variant uses reanalysis weather on the target date and is NOT a "
                "deployable model. It is an upper bound on what perfect weather knowledge "
                "could contribute."
            ),
            "validation_folds": [fold[0] for fold in VALIDATION_FOLDS],
            "final_test_period": f"{TEST_START.date()} to {TEST_END.date()}",
            "max_false_positive_rate": MAX_FALSE_POSITIVE_RATE,
        }, handle, indent=2)

    print("\nHeadroom summary:")
    print(wide.round(3).to_string())
    print(f"\nSaved to {REPORT_DIR}")


if __name__ == "__main__":
    main()
