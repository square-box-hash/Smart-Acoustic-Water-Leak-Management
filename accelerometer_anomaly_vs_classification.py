"""
accelerometer_anomaly_vs_classification.py

Re-tests the Mahalanobis anomaly-score approach on Accelerometer,
this time using the much larger WINDOWED dataset (848 windows, not 80
raw trials), and evaluates it properly with ROC-AUC (how well the
score separates No-Leak from Leak, threshold-independent) rather than
just comparing mean scores by eye.

Directly answers: for Accelerometer specifically, does anomaly-score
framing or multi-class classification work better? We already have
classification's answer (82.9% accuracy, from train_windowed_no_hydrophone.py).
This script gives the anomaly-score side of that comparison on equal
footing (same windowed data, same feature set).

Run: python accelerometer_anomaly_vs_classification.py
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_auc_score, roc_curve

FEATURE_COLS = ["mean", "std", "skewness", "kurtosis", "rms", "dominant_frequency"]


def fit_healthy_reference(healthy_df, feature_cols):
    X = healthy_df[feature_cols].dropna().to_numpy()
    mean_vec = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    cov += np.eye(cov.shape[0]) * 1e-8
    return mean_vec, np.linalg.inv(cov)


def main():
    df = pd.read_pickle("windowed_anomaly_scores.pkl")
    accel = df[(df["sensor"] == "Accelerometer") & (df["leak_type"].notna())].copy()
    accel = accel.dropna(subset=FEATURE_COLS)

    healthy = accel[accel["leak_type"] == "No Leak"]
    mean_vec, inv_cov = fit_healthy_reference(healthy, FEATURE_COLS)

    scores = []
    for _, row in accel.iterrows():
        x = row[FEATURE_COLS].to_numpy(dtype=float)
        scores.append(mahalanobis(x, mean_vec, inv_cov))
    accel["anomaly_score"] = scores

    print(f"Total Accelerometer windows: {len(accel)}")
    print(f"Healthy (No Leak) reference built from {len(healthy)} windows\n")

    # --- Binary separation: No Leak vs ANY leak, via ROC-AUC ---
    accel["is_leak"] = (accel["leak_type"] != "No Leak").astype(int)
    auc_overall = roc_auc_score(accel["is_leak"], accel["anomaly_score"])
    print(f"Overall anomaly-score AUC (No Leak vs Any Leak): {auc_overall:.3f}")
    print("(0.5 = no better than random guessing, 1.0 = perfect separation)\n")

    # --- Per leak-type AUC: does the score separate THIS leak type from
    # No Leak specifically, even if it can't tell leak types apart from
    # each other (which anomaly scoring was never meant to do) ---
    print("Per-leak-type AUC (this leak type vs No Leak only):")
    for leak_type in accel["leak_type"].unique():
        if leak_type == "No Leak":
            continue
        subset = accel[accel["leak_type"].isin([leak_type, "No Leak"])]
        y_true = (subset["leak_type"] == leak_type).astype(int)
        auc = roc_auc_score(y_true, subset["anomaly_score"])
        mean_leak = subset[subset["leak_type"] == leak_type]["anomaly_score"].mean()
        mean_healthy = subset[subset["leak_type"] == "No Leak"]["anomaly_score"].mean()
        direction = "HIGHER" if mean_leak > mean_healthy else "LOWER"
        print(f"  {leak_type:25s} AUC={auc:.3f}  (leak mean={mean_leak:.2f}, "
              f"healthy mean={mean_healthy:.2f}, leak scores {direction})")

    print("\n--- Comparison ---")
    print(f"Classification (multi-class RF, from train_windowed_no_hydrophone.py): 0.829 accuracy")
    print(f"Anomaly score (binary separation, this script):                        {auc_overall:.3f} AUC")
    print("\nNote: these aren't directly the same metric (accuracy vs AUC, "
          "5-class vs binary) -- but both measure how well Accelerometer "
          "data distinguishes leak-related states, just via different "
          "tasks. A high classification accuracy with a mediocre anomaly "
          "AUC would suggest Accelerometer's value is in distinguishing "
          "SPECIFIC leak types (structural signature), not just detecting "
          "generic deviation from normal.")

    accel.to_csv("accelerometer_anomaly_scores.csv", index=False)


if __name__ == "__main__":
    main()