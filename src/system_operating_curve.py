"""Operating curve: is the two-head system better than persistence at *any* budget?

A single 5% false-positive target is arbitrary for a public warning system, and at
that budget the two heads compete for the same allowance.  The informative
question is whether the two-head system dominates the persistence rule across the
whole operating range.

For a grid of false-positive budgets, both systems are calibrated on the prior
winter seasons with the same rule (maximise mean validation recall subject to
FPR <= budget in every fold) and then evaluated once on the held-out period:

* persistence: alert when AQI at issue time is at or above a calibrated level;
* two-head: that rule, plus the clean-origin onset model above its own calibrated
  threshold.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from train_regional_early_warning import (
    CUTOFF,
    HORIZONS,
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

BUDGETS = np.round(np.arange(0.02, 0.21, 0.01), 2)
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


def summarise(observed: np.ndarray, issued: np.ndarray) -> dict:
    tn = int(((~issued) & (observed == 0)).sum())
    fp = int((issued & (observed == 0)).sum())
    fn = int(((~issued) & (observed == 1)).sum())
    tp = int((issued & (observed == 1)).sum())
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

    rows = []
    for horizon in HORIZONS:
        recall_cont, fpr_cont, recall_joint, fpr_joint = [], [], [], []
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

            recall_cont.append(cont_tp / n_pos)
            fpr_cont.append(cont_fp / n_neg)
            recall_joint.append((cont_tp[:, None] + onset_tp[None, :]) / n_pos)
            fpr_joint.append((cont_fp[:, None] + onset_fp[None, :]) / n_neg)

        recall_cont = np.array(recall_cont)
        fpr_cont = np.array(fpr_cont)
        recall_joint = np.array(recall_joint)
        fpr_joint = np.array(fpr_joint)

        test = assemble(frame, horizon, test_predictions)
        test = test[(test.date >= TEST_START) & (test.date <= TEST_END)]
        observed = test.event_observed.to_numpy().astype(int)
        level = test.aqi_at_issue.to_numpy()
        onset = test.onset_probability.fillna(0.0).to_numpy()

        for budget in BUDGETS:
            # Persistence-only configuration.
            worst_cont = fpr_cont.max(axis=0)
            ok = worst_cont <= budget
            if ok.any():
                score = np.where(ok, recall_cont.mean(axis=0), -np.inf)
                index = int(np.argmax(score))
                issued = level >= CONTINUATION_LEVELS[index]
                rows.append({
                    "horizon_days": horizon, "budget": budget, "system": "persistence",
                    "continuation_level": float(CONTINUATION_LEVELS[index]), "onset_threshold": np.nan,
                    "validation_worst_fold_fpr": float(worst_cont[index]),
                    **summarise(observed, issued),
                })

            # Two-head configuration.
            worst_joint = fpr_joint.max(axis=0)
            ok_joint = worst_joint <= budget
            if ok_joint.any():
                score = np.where(ok_joint, recall_joint.mean(axis=0), -np.inf)
                best = np.unravel_index(np.argmax(score), score.shape)
                issued = (level >= CONTINUATION_LEVELS[best[0]]) | (onset >= ONSET_THRESHOLDS[best[1]])
                rows.append({
                    "horizon_days": horizon, "budget": budget, "system": "two-head",
                    "continuation_level": float(CONTINUATION_LEVELS[best[0]]),
                    "onset_threshold": float(ONSET_THRESHOLDS[best[1]]),
                    "validation_worst_fold_fpr": float(worst_joint[best]),
                    **summarise(observed, issued),
                })

    table = pd.DataFrame(rows)
    table.to_csv(REPORT_DIR / "system_operating_curve.csv", index=False)

    with (REPORT_DIR / "system_operating_curve_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "budgets": [float(BUDGETS.min()), float(BUDGETS.max())],
            "calibration": "per budget, maximise mean validation recall subject to FPR <= budget in every fold",
            "final_test_period": f"{TEST_START.date()} to {TEST_END.date()}",
            "note": "Persistence is parametrised by the continuation level; two-head adds the onset model.",
        }, handle, indent=2)

    for horizon in HORIZONS:
        block = table[table.horizon_days.eq(horizon)]
        wide = block.pivot(index="budget", columns="system", values="Recall")
        wide.columns = [f"recall_{column}" for column in wide.columns]
        print(f"\nt+{horizon}: test recall by validation-selected budget")
        print(wide.round(3).to_string())
        wf = block.pivot(index="budget", columns="system", values="false_positive_rate")
        print("achieved test FPR:")
        print(wf.round(3).to_string())

    # Operating-curve figure.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = ROOT / "reports" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(HORIZONS), figsize=(5.0 * len(HORIZONS), 4.3), sharey=True)
    for axis, horizon in zip(np.atleast_1d(axes), HORIZONS):
        block = table[table.horizon_days.eq(horizon)]
        for system, style in (("persistence", "o--"), ("two-head", "s-")):
            subset = block[block.system.eq(system)].sort_values("budget")
            axis.plot(subset.budget, subset.Recall, style, label=system, markersize=4, linewidth=1.8)
        axis.set_title(f"t+{horizon}")
        axis.set_xlabel("Validation false-alarm budget")
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Test recall (AQI > 200)")
    fig.suptitle("Warning recall against the false-alarm budget, held-out Nov 2024 - Apr 2025")
    fig.tight_layout()
    fig.savefig(figure_dir / "system_operating_curve.png", dpi=160)
    plt.close(fig)

    print(f"\nSaved to {REPORT_DIR / 'system_operating_curve.csv'}")


if __name__ == "__main__":
    main()
