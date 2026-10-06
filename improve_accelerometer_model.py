"""
improve_accelerometer_model.py

Two improvements to the Accelerometer classifier, no new data needed:

1. K-FOLD CROSS-VALIDATION: instead of one 80/20 split (which can be
   lucky or unlucky), run 5-fold stratified CV and report mean +/- std
   accuracy. This is what several of the reviewed papers do (e.g.
   FiT-WST+ reports std across repeated runs) and gives a more honest,
   defensible accuracy estimate than a single split.

2. TRIAL-LEVEL MAJORITY VOTING: each 30s trial produces several
   overlapping windows, each scored independently so far. A real
   deployment cares about "what's the verdict for this recording", not
   "what's the verdict for this one 3-second slice". Aggregating each
   trial's window predictions by majority vote is standard practice and
   typically smooths out noisy individual-window errors -- this script
   reports BOTH window-level and trial-level accuracy so the
   improvement from aggregation is visible and quantified, not assumed.

Run: python improve_accelerometer_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from collections import Counter

RANDOM_STATE = 42
FEATURE_COLS = ["mean", "std", "skewness", "kurtosis", "rms", "dominant_frequency"]
N_FOLDS = 5


def majority_vote(preds):
    """Returns the most common prediction; ties broken by first-seen order."""
    return Counter(preds).most_common(1)[0][0]


def main():
    df = pd.read_pickle("windowed_v2.pkl")
    accel = df[(df["sensor"] == "Accelerometer") & (df["leak_type"].notna())].copy()
    accel = accel.dropna(subset=FEATURE_COLS)

    X = accel[FEATURE_COLS]
    y = accel["leak_type"]

    print(f"Total windows: {len(accel)}")
    print(f"Class distribution:\n{y.value_counts()}\n")

    # --- Part 1: K-fold cross-validation ---
    # IMPORTANT: grouped by trial_file, not plain StratifiedKFold, for
    # the same reason as the majority-voting split below -- with 90%
    # window overlap, windows from the same trial are near-duplicates,
    # so a plain stratified split can leak near-identical windows across
    # the train/test boundary and inflate accuracy. StratifiedGroupKFold
    # keeps every window from one trial entirely within one fold.
    print("=" * 60)
    print(f"{N_FOLDS}-FOLD CROSS-VALIDATION (grouped by trial, window-level scoring)")
    print("=" * 60)

    from sklearn.model_selection import StratifiedGroupKFold

    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced")
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(clf, X, y, cv=sgkf, groups=accel["trial_file"], scoring="accuracy")

    print(f"Fold accuracies: {[f'{s:.3f}' for s in scores]}")
    print(f"Mean accuracy: {scores.mean():.3f}  +/-  {scores.std():.3f}")
    print("(This is a more reliable estimate than a single train/test "
          "split, since it's averaged across 5 different, non-overlapping "
          "test sets rather than depending on one lucky/unlucky split.)")

    # --- Part 2: trial-level majority voting ---
    # IMPORTANT: with 90% window overlap, consecutive windows from the
    # same trial are nearly identical (they share ~90% of their samples).
    # Splitting by WINDOW (as done naively before) can put near-duplicate
    # windows from the same trial on both sides of the train/test split
    # -- this is data leakage: the model effectively sees a near-copy of
    # a "test" trial during training, inflating accuracy artificially.
    # The correct approach is to split by TRIAL: every window from one
    # recording goes entirely into train OR entirely into test, never
    # both.
    print("\n" + "=" * 60)
    print("TRIAL-LEVEL MAJORITY VOTING (fixed: split by trial, not window)")
    print("=" * 60)

    from sklearn.model_selection import train_test_split

    # One label per trial (all windows from a trial share the same
    # leak_type), used only to stratify the trial-level split.
    trial_labels = accel.groupby("trial_file")["leak_type"].first()
    train_trials, test_trials = train_test_split(
        trial_labels.index, test_size=0.20, random_state=RANDOM_STATE,
        stratify=trial_labels.values,
    )

    train_mask = accel["trial_file"].isin(train_trials)
    test_mask = accel["trial_file"].isin(test_trials)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train trials: {len(train_trials)}  |  Test trials: {len(test_trials)}")
    print(f"Train windows: {len(X_train)}  |  Test windows: {len(X_test)}")

    clf2 = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced")
    clf2.fit(X_train, y_train)
    window_preds = clf2.predict(X_test)

    window_acc = accuracy_score(y_test, window_preds)
    print(f"Window-level accuracy (baseline, no aggregation): {window_acc:.3f}")

    test_df = accel[test_mask].copy()
    test_df["predicted"] = window_preds
    test_df["actual"] = y_test.values

    # Aggregate: one row per trial, majority-vote its windows' predictions
    trial_results = []
    for trial_file, group in test_df.groupby("trial_file"):
        actual = group["actual"].iloc[0]  # all windows from one trial share the same true label
        predicted = majority_vote(group["predicted"].tolist())
        trial_results.append({"trial_file": trial_file, "actual": actual,
                               "predicted": predicted, "n_windows": len(group)})

    trial_df = pd.DataFrame(trial_results)
    trial_acc = accuracy_score(trial_df["actual"], trial_df["predicted"])

    print(f"Trial-level accuracy (majority vote across each trial's windows): {trial_acc:.3f}")
    print(f"Improvement from aggregation: {trial_acc - window_acc:+.3f}")
    print(f"\nNumber of test trials aggregated: {len(trial_df)}")
    print(f"Average windows per trial in test set: {trial_df['n_windows'].mean():.1f}")

    misclassified_trials = trial_df[trial_df["actual"] != trial_df["predicted"]]
    print(f"\nMisclassified trials ({len(misclassified_trials)}):")
    print(misclassified_trials.to_string(index=False))

    trial_df.to_csv("trial_level_predictions.csv", index=False)


if __name__ == "__main__":
    main()