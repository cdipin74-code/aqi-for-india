"""Regenerate PR and ROC curves with explicit no-skill reference lines.

Every precision-recall panel carries two references so the reader cannot mistake
raw PR-AUC for skill:

* the test-period prevalence, which is the PR-AUC of a random score;
* the persistence score (AQI at issue time), which is the strongest trivial
  predictor for a highly autocorrelated target.

ROC panels carry the 0.5 diagonal.  Curves use the raw (uncalibrated) model
probabilities because Platt scaling is monotone and leaves PR-AUC and ROC-AUC
unchanged.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from train_regional_early_warning import DATA_PATH, ROOT

REPORT_DIR = ROOT / "reports" / "ablation"
FIGURE_DIR = ROOT / "reports" / "figures"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    ("AQI_gt_200_t_plus_1", "AQI > 200, t+1"),
    ("AQI_gt_200_t_plus_2", "AQI > 200, t+2"),
    ("AQI_gt_200_t_plus_3", "AQI > 200, t+3"),
    ("AQI_gt_300_t_plus_1", "AQI > 300, t+1"),
    ("AQI_gt_300_t_plus_2", "AQI > 300, t+2"),
    ("AQI_gt_300_t_plus_3", "AQI > 300, t+3"),
]


def load_issue_aqi() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, usecols=["date", "state", "area", "aqi_value"], parse_dates=["date"])
    data["location_key"] = data["state"].str.strip() + "||" + data["area"].str.strip()
    return data[["date", "location_key", "aqi_value"]].drop_duplicates(["date", "location_key"])


def main() -> None:
    predictions = pd.read_csv(
        ROOT / "reports" / "calibration" / "calibrated_alert_test_predictions.csv", parse_dates=["date"]
    )
    predictions = predictions[predictions.policy.eq("Balanced F1 (tuning)")]
    issue_aqi = load_issue_aqi()
    rows = []

    for task, title in TASKS:
        frame = predictions[predictions.task.eq(task)].merge(issue_aqi, on=["date", "location_key"], how="left")
        y = frame.event_observed.to_numpy()
        model_score = frame.raw_probability.to_numpy()
        persistence = frame.aqi_value.to_numpy()
        prevalence = float(y.mean())

        model_pr = average_precision_score(y, model_score)
        persistence_pr = average_precision_score(y, persistence)
        model_roc = roc_auc_score(y, model_score)
        persistence_roc = roc_auc_score(y, persistence)

        rows.append({
            "task": task, "title": title, "n_rows": len(frame), "n_events": int(y.sum()),
            "prevalence": prevalence,
            "model_PR_AUC": model_pr, "persistence_PR_AUC": persistence_pr,
            "PR_AUC_gain_over_persistence": model_pr - persistence_pr,
            "model_ROC_AUC": model_roc, "persistence_ROC_AUC": persistence_roc,
        })

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

        precision, recall, _ = precision_recall_curve(y, model_score)
        p_precision, p_recall, _ = precision_recall_curve(y, persistence)
        axes[0].plot(recall, precision, lw=2, label=f"XGBoost (PR-AUC {model_pr:.3f})")
        axes[0].plot(p_recall, p_precision, lw=1.6, ls="--", label=f"Persistence score (PR-AUC {persistence_pr:.3f})")
        axes[0].axhline(prevalence, color="black", lw=1.2, ls=":", label=f"No skill / prevalence ({prevalence:.3f})")
        axes[0].set_xlabel("Recall")
        axes[0].set_ylabel("Precision")
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1.02)
        axes[0].set_title(f"Precision-recall: {title}")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].grid(alpha=0.25)

        fpr, tpr, _ = roc_curve(y, model_score)
        p_fpr, p_tpr, _ = roc_curve(y, persistence)
        axes[1].plot(fpr, tpr, lw=2, label=f"XGBoost (ROC-AUC {model_roc:.3f})")
        axes[1].plot(p_fpr, p_tpr, lw=1.6, ls="--", label=f"Persistence score (ROC-AUC {persistence_roc:.3f})")
        axes[1].plot([0, 1], [0, 1], color="black", lw=1.2, ls=":", label="No skill (0.5)")
        axes[1].set_xlabel("False positive rate")
        axes[1].set_ylabel("True positive rate")
        axes[1].set_title(f"ROC: {title}")
        axes[1].legend(loc="lower right", fontsize=8)
        axes[1].grid(alpha=0.25)

        fig.tight_layout()
        safe_name = task.replace(" ", "_")
        fig.savefig(FIGURE_DIR / f"pr_roc_with_baselines_{safe_name}.png", dpi=160)
        plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(REPORT_DIR / "pr_auc_versus_persistence.csv", index=False)
    print(table.round(3).to_string(index=False))
    print(f"\nFigures written to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
