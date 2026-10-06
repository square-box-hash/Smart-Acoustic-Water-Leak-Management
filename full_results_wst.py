"""
Full result details for the CWT (Morlet scale-power) + SVM binary pipeline in
wst_svm_binary_loto.py. Trial-level Leave-One-Trial-Out CV.

Training uses the windows of all OTHER trials; each held-out trial is classified
by majority vote of its windows (same rule as your original script, including
the tie rule: an exact 50/50 vote rounds to No-Leak).

Put this file in the SAME folder as wst_svm_binary_loto.py and trials.pkl.
Run:  python full_results_wst.py
"""
import os
import time
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from wst_svm_binary_loto import (
    RANDOM_STATE,
    SENSOR,
    TRIALS_PKL,
    WINDOW_LEN,
    build_wst_extractor,
    load_and_window_trials,
)

# ---------------- CONFIG ----------------
# Set CLASS_WEIGHT to the "BEST" weights your original script printed.
CLASS_WEIGHT = {0: 4, 1: 1}      # 0 = No-Leak, 1 = Leak
WEIGHT_OPTIONS = [{0: 1, 1: 1}, {0: 2, 1: 1}, {0: 3, 1: 1}, {0: 4, 1: 1}, "balanced"]
N_PERM = 100                     # permutation repeats (each ~ one full LOTO run)
CACHE = "wst_feature_cache.npz"  # features are cached here after the first run
OUT_CSV = "loto_predictions.csv"
SEED = 0
# ----------------------------------------


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def get_features():
    windows_df = load_and_window_trials()
    ids = windows_df["trial_id"].values.astype(str)
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if len(z["trial_ids"]) == len(ids) and (z["trial_ids"] == ids).all():
            print(f"Loaded cached features from {CACHE}")
            return windows_df, z["features"]
    extract = build_wst_extractor(WINDOW_LEN)
    print("Extracting features for all windows (first run only)...")
    features = np.stack([extract(w) for w in windows_df["window_signal"]])
    np.savez(CACHE, features=features, trial_ids=ids)
    return windows_df, features


def loto(features, y_win, trial_ids, trials, weights):
    """Return per-trial: predicted label, fraction of windows voting Leak, mean score."""
    pred, vote, score = [], [], []
    for t in trials:
        te = trial_ids == t
        tr = ~te
        sc = StandardScaler()
        Xtr = sc.fit_transform(features[tr])
        Xte = sc.transform(features[te])
        clf = SVC(kernel="rbf", class_weight=weights, random_state=RANDOM_STATE)
        clf.fit(Xtr, y_win[tr])
        p = clf.predict(Xte)
        vote.append(float(p.mean()))
        pred.append(int(round(float(p.mean()))))
        score.append(float(clf.decision_function(Xte).mean()))
    return np.array(pred), np.array(vote), np.array(score)


def main():
    windows_df, features = get_features()
    trial_ids = windows_df["trial_id"].values.astype(str)
    y_win = windows_df["binary_label"].values.astype(int)

    trial_tbl = (
        windows_df.assign(trial_id=trial_ids)
        .groupby("trial_id")
        .agg(leak_type=("leak_type", "first"), y=("binary_label", "first"),
             n_windows=("binary_label", "size"))
        .reset_index()
    )
    trials = trial_tbl["trial_id"].values
    y = trial_tbl["y"].values.astype(int)
    n, n0, n1 = len(y), int((y == 0).sum()), int((y == 1).sum())

    pred, vote, score = loto(features, y_win, trial_ids, trials, CLASS_WEIGHT)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    print("\n=== SETTINGS ===")
    print(f"sensor={SENSOR}, class_weight={CLASS_WEIGHT}, windows={len(windows_df)}")
    print(f"trials: {n} (No-Leak={n0}, Leak={n1})")
    print(f"majority-class baseline accuracy: {max(n0, n1) / n:.3f}")

    print("\n=== HEADLINE METRICS (trial-level LOTO-CV) ===")
    lo, hi = wilson(int((pred == y).sum()), n)
    print(f"accuracy:          {(pred == y).mean():.3f}  (95% Wilson CI {lo:.3f}-{hi:.3f})")
    print(f"balanced accuracy: {balanced_accuracy_score(y, pred):.3f}")
    print(f"MCC:               {matthews_corrcoef(y, pred):.3f}")
    print(f"ROC-AUC:           {roc_auc_score(y, score):.3f}  (mean window score per trial)")
    print(f"PR-AUC (Leak):     {average_precision_score(y, score):.3f}")

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

    print("\n=== VOTE CONFIDENCE ===")
    amb = int(((vote >= 0.35) & (vote <= 0.65)).sum())
    ties = int((vote == 0.5).sum())
    print(f"trials with a close vote (35-65% of windows voting Leak): {amb} of {n}")
    print(f"exact 50/50 ties (resolved to No-Leak): {ties}")

    print("\n=== BREAKDOWN BY METADATA (check for confounds) ===")
    tmp = trial_tbl.set_index("trial_id").copy()
    tmp["pred"] = pred
    tmp["vote_leak"] = vote
    tmp["correct"] = (tmp["y"] == tmp["pred"]).astype(int)
    try:
        raw = pd.read_pickle(TRIALS_PKL)
        raw = raw[raw["sensor"] == SENSOR].drop_duplicates("raw_name")
        meta_cols = []
        for c in raw.columns:
            if c in ("signal", "raw_name", "sensor") or c in tmp.columns:
                continue
            try:
                if raw[c].nunique() <= 12:
                    meta_cols.append(c)
            except TypeError:
                pass
        meta = raw.set_index(raw["raw_name"].astype(str))[meta_cols]
        tmp = tmp.join(meta)
    except Exception as e:  # metadata is optional
        print(f"(could not merge extra metadata: {e})")
        meta_cols = []
    for col in ["leak_type"] + meta_cols:
        print(f"\n-- by {col} --")
        print(tmp.groupby(col, dropna=False).agg(
            n=("y", "size"),
            true_leak=("y", "sum"),
            predicted_leak=("pred", "sum"),
            accuracy=("correct", "mean"),
            mean_vote_leak=("vote_leak", "mean"),
        ).round(3))

    print("\n=== CLASS-WEIGHT SENSITIVITY ===")
    print("Weights were chosen after seeing results; the original selection rule was")
    print("0.5*accuracy + 0.5*No-Leak recall. Report this table openly.")
    rows = []
    for w in WEIGHT_OPTIONS:
        pw, _, _ = loto(features, y_win, trial_ids, trials, w)
        acc = float((pw == y).mean())
        r0 = float(recall_score(y, pw, pos_label=0, zero_division=0))
        rows.append({
            "class_weight": str(w),
            "accuracy": round(acc, 3),
            "balanced_acc": round(balanced_accuracy_score(y, pw), 3),
            "recall_NoLeak": round(r0, 3),
            "recall_Leak": round(float(recall_score(y, pw, pos_label=1, zero_division=0)), 3),
            "selection_score": round(0.5 * acc + 0.5 * r0, 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n=== PERMUTATION TEST (balanced accuracy, {N_PERM} trial-label shuffles) ===")
    rng = np.random.default_rng(SEED)
    obs = balanced_accuracy_score(y, pred)
    null = []
    t0 = time.time()
    for i in range(N_PERM):
        yp = rng.permutation(y)
        yp_win = pd.Series(yp, index=trials).reindex(trial_ids).values.astype(int)
        pp, _, _ = loto(features, yp_win, trial_ids, trials, CLASS_WEIGHT)
        null.append(balanced_accuracy_score(yp, pp))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_PERM} done ({time.time() - t0:.0f}s)")
    null = np.array(null)
    p_val = (1 + np.sum(null >= obs)) / (N_PERM + 1)
    print(f"observed balanced accuracy={obs:.3f}, null mean={null.mean():.3f}, "
          f"null 95th pct={np.percentile(null, 95):.3f}, p={p_val:.4f}")

    out = trial_tbl.copy()
    out["y_pred"] = pred
    out["vote_leak_fraction"] = vote
    out["mean_score"] = score
    out.to_csv(OUT_CSV, index=False)
    print(f"\nPer-trial predictions saved to {OUT_CSV}")


if __name__ == "__main__":
    main()