"""
loto_cv.py

Leave-One-Trial-Out Cross-Validation (LOTO-CV) for Accelerometer.

With only 80 independent trials, a single 80/20 split or even 5-fold CV
is noisy (we saw fold accuracies swing from 22.8% to 45.4% in the
5-fold run). LOTO-CV squeezes the most reliable possible estimate out
of a small trial count: train on 79 trials, test on the 1 remaining
trial, repeat so every trial is the test case exactly once (80 total
train/test cycles). The final accuracy is the average across all 80
held-out trials -- every single trial contributes to the test set
exactly once, and no trial's windows ever appear in both train and
test for its own fold (same leakage-prevention principle as before).

This is intentionally still using the SAME simple feature set (mean,
std, skewness, kurtosis, RMS, dominant frequency) as the baseline we've
already tested -- this script's job is to give an honest, reliable
number for THAT feature set specifically, to compare against once WST
features are added next.

Run: python loto_cv.py
(This will take a while -- 80 separate model trainings.)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

RANDOM_STATE = 42
FEATURE_COLS = ["mean", "std", "skewness", "kurtosis", "rms", "dominant_frequency"]


def majority_vote(preds):
    return Counter(preds).most_common(1)[0][0]


def main():
    df = pd.read_pickle("windowed_v2.pkl")
    accel = df[(df["sensor"] == "Accelerometer") & (df["leak_type"].notna())].copy()
    accel = accel.dropna(subset=FEATURE_COLS)

    trial_files = accel["trial_file"].unique()
    print(f"Total trials: {len(trial_files)}")
    print(f"Total windows: {len(accel)}\n")

    results = []
    for i, held_out_trial in enumerate(trial_files):
        train_mask = accel["trial_file"] != held_out_trial
        test_mask = accel["trial_file"] == held_out_trial

        X_train = accel.loc[train_mask, FEATURE_COLS]
        y_train = accel.loc[train_mask, "leak_type"]
        X_test = accel.loc[test_mask, FEATURE_COLS]
        y_test = accel.loc[test_mask, "leak_type"]

        actual = y_test.iloc[0]  # all windows from this trial share the true label

        clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight="balanced")
        clf.fit(X_train, y_train)

        window_preds = clf.predict(X_test)
        window_acc = (window_preds == actual).mean()
        trial_pred = majority_vote(window_preds.tolist())

        results.append({
            "trial_file": held_out_trial,
            "actual": actual,
            "predicted": trial_pred,
            "correct": trial_pred == actual,
            "window_accuracy_within_trial": window_acc,
            "n_windows": len(X_test),
        })

        if (i + 1) % 10 == 0:
            print(f"  ...completed {i + 1}/{len(trial_files)} trials")

    results_df = pd.DataFrame(results)
    results_df.to_csv("loto_cv_results.csv", index=False)

    overall_acc = results_df["correct"].mean()
    print(f"\n{'='*60}")
    print(f"LOTO-CV TRIAL-LEVEL ACCURACY: {overall_acc:.3f}  "
          f"({results_df['correct'].sum()} / {len(results_df)} trials correct)")
    print(f"{'='*60}")

    print("\nAccuracy by leak type:")
    print(results_df.groupby("actual")["correct"].agg(["sum", "count", "mean"]))

    print("\nConfusion pattern (actual -> predicted, for misclassified trials only):")
    misclassified = results_df[~results_df["correct"]]
    print(misclassified.groupby(["actual", "predicted"]).size().sort_values(ascending=False))

    print(f"\nFull results saved to loto_cv_results.csv")
    print(f"\nCompare this LOTO-CV number against:")
    print(f"  - Trial-level 80/20 split (earlier):     0.337")
    print(f"  - Within-trial time-split (diagnostic):  0.722 (NOT comparable -- different question)")


if __name__ == "__main__":
    main()