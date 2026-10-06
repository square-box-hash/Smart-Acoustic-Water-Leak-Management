"""
train_model.py

Trains a Random Forest classifier on the extracted feature table to
distinguish leak vs. no-leak (and separately, leak TYPE among leak
trials) from the acoustic features computed in extract_features.py.

Beyond just reporting accuracy, this script also interrogates the
model:
  - feature importances (which features actually mattered)
  - confusion matrix (what gets confused with what)
  - which specific trials were misclassified (useful for the logbook --
    "why" a specific trial failed is often more interesting than the
    headline accuracy number)

Run: python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
)

RANDOM_STATE = 42

# Features to actually feed the model. We use the RAW per-trial features
# (not baseline-relative ones) here for the primary leak-vs-no-leak task,
# because in real deployment you won't always have a matching "No Leak"
# baseline trial for comparison on demand -- the model should learn to
# recognize a leak from the signal's own characteristics, not require a
# baseline lookup at prediction time. Baseline-relative features are kept
# in the table for exploratory analysis, not as model inputs.
FEATURE_COLS = [
    "dominant_frequency",
    "total_energy",
    "rms_amplitude",
    "signal_variance",
    "spectral_entropy",
    "band_energy_1700_2000",
]


def train_and_evaluate(df, label_col, task_name):
    print(f"\n{'='*60}\nTask: {task_name}\n{'='*60}")

    data = df.dropna(subset=FEATURE_COLS + [label_col])
    X = data[FEATURE_COLS]
    y = data[label_col]

    print(f"Samples: {len(data)}  |  Classes: {sorted(y.unique())}")
    print(f"Class distribution:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion matrix (rows=actual, cols=predicted):")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    print("\nFeature importances:")
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS)
    print(importances.sort_values(ascending=False))

    # Show misclassified trials with their identifying info -- useful
    # for the logbook: "why did this specific trial fail?"
    test_meta = data.loc[X_test.index, ["sensor", "network", "channel", "state", "leak_type"]]
    misclassified = test_meta.copy()
    misclassified["actual"] = y_test.values
    misclassified["predicted"] = y_pred
    misclassified = misclassified[misclassified["actual"] != misclassified["predicted"]]
    if len(misclassified) > 0:
        print(f"\n{len(misclassified)} misclassified trial(s):")
        print(misclassified.to_string(index=False))
    else:
        print("\nNo misclassified trials in the test set.")

    return clf, acc


def main():
    df = pd.read_pickle("trials_with_features.pkl")

    # Task 1: binary leak vs. no-leak
    df["is_leak"] = (df["leak_type"] != "No Leak").map({True: "Leak", False: "No Leak"})
    train_and_evaluate(df, "is_leak", "Leak vs No Leak (binary)")

    # Task 2: multi-class leak TYPE, leak trials only (excludes No Leak
    # rows, since the question here is "which kind of leak is it",
    # assuming we already know a leak is present)
    leak_only = df[df["leak_type"] != "No Leak"].copy()
    train_and_evaluate(leak_only, "leak_type", "Leak Type Classification (multi-class)")


if __name__ == "__main__":
    main()