"""
Full result details for the frozen wavelet + SVM binary (Leak vs No-Leak) pipeline.
Trial-level Leave-One-Trial-Out CV (LOTO-CV).

Edit get_data() so it returns YOUR wavelet features. Everything else is evaluation.

Run:  python full_results_wavelet.py
"""
from math import sqrt

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------- CONFIG: copy the SAME settings as your 82% run ----------
KERNEL = "rbf"
C = 1.0
GAMMA = "scale"
CLASS_WEIGHT = {0: 4.0, 1: 1.0}     # 0 = No-Leak, 1 = Leak
N_PERM = 500                        # permutation test repeats
WEIGHT_GRID = [1, 2, 3, 4, 5, 6]    # No-Leak weights for the sensitivity table
OUT_CSV = "loto_predictions.csv"
SEED = 0
# --------------------------------------------------------------------


def get_data():
    """Return X, y, groups, meta.

    X:      (n_trials, n_features) wavelet features from YOUR pipeline
    y:      (n_trials,) 1 = Leak, 0 = No-Leak
    groups: (n_trials,) trial id; each trial is held out as one unit
    meta:   DataFrame, one row per trial, with columns such as
            leak_type, network, condition (any you have)

    Example (edit to match your code):
        df = pd.read_pickle("trials.pkl")
        df = df[df["sensor"] == "Accelerometer"].reset_index(drop=True)
        X = np.vstack([your_wavelet_features(s) for s in df["signal"]])
        norm = df["leak_type"].str.lower().str.replace(r"[ _-]", "", regex=True)
        y = (~norm.str.startswith("noleak")).astype(int).values
        groups = np.arange(len(df))
        meta = df[["leak_type"]].copy()
        return X, y, groups, meta
    """
    raise NotImplementedError("Edit get_data() to return X, y, groups, meta")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def make_clf(w0=None):
    cw = CLASS_WEIGHT if w0 is None else {0: float(w0), 1: 1.0}
    return make_pipeline(
        StandardScaler(),
        SVC(kernel=KERNEL, C=C, gamma=GAMMA, class_weight=cw),
    )


def loto(X, y, groups, w0=None, want_scores=False):
    cv = LeaveOneGroupOut()
    clf = make_clf(w0)
    pred = cross_val_predict(clf, X, y, groups=groups, cv=cv)
    scores = None
    if want_scores:
        # Note: each fold has its own model, so scores are not strictly comparable
        scores = cross_val_predict(clf, X, y, groups=groups, cv=cv,
                                   method="decision_function")
    return pred, scores


def main():
    X, y, groups, meta = get_data()
    y = np.asarray(y).astype(int)
    n = len(y)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())

    pred, scores = loto(X, y, groups, want_scores=True)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    print("=== SETTINGS ===")
    print(f"kernel={KERNEL}, C={C}, gamma={GAMMA}, class_weight={CLASS_WEIGHT}")
    print(f"trials: {n}  (No-Leak={n0}, Leak={n1})")
    print(f"majority-class baseline accuracy: {max(n0, n1) / n:.3f}")

    print("\n=== HEADLINE METRICS (trial-level LOTO-CV) ===")
    acc = (pred == y).mean()
    lo, hi = wilson(int((pred == y).sum()), n)
    print(f"accuracy:          {acc:.3f}  (95% Wilson CI {lo:.3f}-{hi:.3f})")
    print(f"balanced accuracy: {balanced_accuracy_score(y, pred):.3f}")
    print(f"MCC:               {matthews_corrcoef(y, pred):.3f}")
    print(f"ROC-AUC:           {roc_auc_score(y, scores):.3f}")
    print(f"PR-AUC (Leak):     {average_precision_score(y, scores):.3f}")

    print("\n=== PER-CLASS ===")
    for name, lab, k, tot in [("No-Leak", 0, tn, n0), ("Leak", 1, tp, n1)]:
        lo, hi = wilson(int(k), tot)
        print(
            f"{name}: recall={recall_score(y, pred, pos_label=lab):.3f} "
            f"(95% CI {lo:.3f}-{hi:.3f}), "
            f"precision={precision_score(y, pred, pos_label=lab, zero_division=0):.3f}, "
            f"F1={f1_score(y, pred, pos_label=lab, zero_division=0):.3f}"
        )

    print("\n=== CONFUSION MATRIX (rows = true, cols = predicted) ===")
    print(pd.DataFrame([[tn, fp], [fn, tp]],
                       index=["true No-Leak", "true Leak"],
                       columns=["pred No-Leak", "pred Leak"]))

    print("\n=== PERMUTATION TEST (balanced accuracy) ===")
    rng = np.random.default_rng(SEED)
    obs = balanced_accuracy_score(y, pred)
    null = []
    for _ in range(N_PERM):
        yp = rng.permutation(y)
        pp, _ = loto(X, yp, groups)
        null.append(balanced_accuracy_score(yp, pp))
    null = np.array(null)
    p_val = (1 + np.sum(null >= obs)) / (N_PERM + 1)
    print(f"observed={obs:.3f}, null mean={null.mean():.3f}, "
          f"null 95th pct={np.percentile(null, 95):.3f}, p={p_val:.4f} "
          f"({N_PERM} permutations)")

    print("\n=== BREAKDOWN BY METADATA (check for confounds) ===")
    tmp = meta.reset_index(drop=True).copy()
    tmp["y"] = y
    tmp["pred"] = pred
    tmp["correct"] = (tmp["y"] == tmp["pred"]).astype(int)
    for col in meta.columns:
        print(f"\n-- by {col} --")
        print(tmp.groupby(col).agg(
            n=("y", "size"),
            true_leak=("y", "sum"),
            predicted_leak=("pred", "sum"),
            accuracy=("correct", "mean"),
        ).round(3))

    print("\n=== CLASS-WEIGHT SENSITIVITY (No-Leak weight; Leak weight = 1) ===")
    print("Weights were chosen after seeing results, so report this table too.")
    rows = []
    for w in WEIGHT_GRID:
        pw, _ = loto(X, y, groups, w0=w)
        rows.append({
            "w_NoLeak": w,
            "accuracy": round((pw == y).mean(), 3),
            "balanced_acc": round(balanced_accuracy_score(y, pw), 3),
            "recall_NoLeak": round(recall_score(y, pw, pos_label=0), 3),
            "recall_Leak": round(recall_score(y, pw, pos_label=1), 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    out = meta.reset_index(drop=True).copy()
    out["group"] = np.asarray(groups)
    out["y_true"] = y
    out["y_pred"] = pred
    out["score"] = scores
    out.to_csv(OUT_CSV, index=False)
    print(f"\nPer-trial predictions saved to {OUT_CSV}")


if __name__ == "__main__":
    main()