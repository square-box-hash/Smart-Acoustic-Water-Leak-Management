"""
train_windowed_no_hydrophone.py

Uses the windowed feature table (built in anomaly_pipeline.py) to train
proper multi-class Random Forest classifiers with MUCH more data per
class than our original one-vector-per-trial attempt (282 trials -> now
thousands of windows). Compares:
  1. Accelerometer only
  2. Dynamic Pressure Sensor only
  3. Hydrophone only            (for reference/comparison)
  4. Accelerometer + Dynamic Pressure Sensor combined (the "no
     hydrophone" test requested)
  5. All three sensors combined (for direct before/after comparison)

This directly tests whether dropping Hydrophone -- which we've now
found unreliable both as an anomaly-score contributor and as a simple
verifier -- actually improves classification accuracy, or whether it
was harmless/neutral all along.

Run: python train_windowed_no_hydrophone.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42
FEATURE_COLS = ["mean", "std", "skewness", "kurtosis", "rms", "dominant_frequency"]


def train_and_report(df, label, feature_cols=FEATURE_COLS):
    print(f"\n{'='*60}\n{label}\n{'='*60}")

    data = df.dropna(subset=feature_cols + ["leak_type"])
    # Windows carry a "sensor" column with a categorical one-hot when
    # multiple sensors are combined, so the model can (if useful) learn
    # sensor-specific patterns rather than assuming all sensors behave
    # identically.
    if data["sensor"].nunique() > 1:
        sensor_dummies = pd.get_dummies(data["sensor"], prefix="sensor")
        X = pd.concat([data[feature_cols].reset_index(drop=True),
                       sensor_dummies.reset_index(drop=True)], axis=1)
    else:
        X = data[feature_cols]

    y = data["leak_type"]

    print(f"Samples: {len(data)}  |  Classes: {sorted(y.unique())}")
    print(f"Class distribution:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    return acc


def main():
    df = pd.read_pickle("windowed_anomaly_scores.pkl")
    df = df[df["leak_type"].notna()].copy()

    results = {}

    # 1-3: single sensors
    for sensor in ["Accelerometer", "DynamicPressureSensor", "Hydrophone"]:
        sub = df[df["sensor"] == sensor]
        results[sensor] = train_and_report(sub, f"{sensor} ONLY")

    # 4: Accelerometer + Dynamic Pressure Sensor (no hydrophone)
    sub = df[df["sensor"].isin(["Accelerometer", "DynamicPressureSensor"])]
    results["Accel + DPS (no Hydrophone)"] = train_and_report(
        sub, "Accelerometer + Dynamic Pressure Sensor (NO HYDROPHONE)"
    )

    # 5: all three sensors combined, for direct before/after comparison
    results["All three sensors"] = train_and_report(df, "ALL THREE SENSORS COMBINED")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for label, acc in results.items():
        print(f"{label:40s}: {acc:.3f}")


if __name__ == "__main__":
    main()