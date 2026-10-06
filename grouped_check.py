"""
Second check: leave-one-EXPERIMENT-out instead of leave-one-recording-out.

Your 80 recordings look like 40 experiments (network x flow state x leak type),
each recorded by two accelerometers (A1, A2). In leave-one-recording-out, the
partner channel of the held-out experiment stays in the training set. This script
holds out BOTH channels of an experiment together.

Put this file in the same folder as full_results_wst.py.
Run:  python grouped_check.py
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from full_results_wst import CLASS_WEIGHT, get_features, wilson
from wst_svm_binary_loto import RANDOM_STATE, SENSOR, TRIALS_PKL

# ---------------- CONFIG ----------------
GROUP_COLS = ["network", "state", "leak_type"]   # defines one experiment
N_PERM = 100
SEED = 0
OUT_CSV = "grouped_predictions.csv"
# ----------------------------------------


def build_table(windows_df):
    ids = windows_df["trial_id"].values.astype(str)
    tbl = (
        windows_df.assign(trial_id=ids)
        .groupby("trial_id")
        .agg(leak_type=("leak_type", "first"), y=("binary_label", "first"))
        .reset_index()
    )
    raw = pd.read_pickle(TRIALS_PKL)
    raw = raw[raw["sensor"] == SENSOR].drop_duplicates("raw_name").copy()
    raw["trial_id"] = raw["raw_name"].astype(str)
    extra = [c for c in ["network", "state", "channel"] if c in raw.columns]
    tbl = tbl.merge(raw[["trial_id"] + extra], on="trial_id", how="left")
    tbl["experiment"] = tbl[GROUP_COLS].astype(str).agg("|".join, axis=1)
    return tbl


def loto_grouped(features, y_win, trial_ids, win_groups, trials, weights):
    vote_d, score_d = {}, {}
    for g in np.unique(win_groups):
        te = win_groups == g
        tr = ~te
        sc = StandardScaler()
        Xtr = sc.fit_transform(features[tr])
        Xte = sc.transform(features[te])
        clf = SVC(kernel="rbf", class_weight=weights, random_state=RANDOM_STATE)
        clf.fit(Xtr, y_win[tr])
        p = clf.predict(Xte)
        s = clf.decision_function(Xte)
        ids_te = trial_ids[te]
        for t in np.unique(ids_te):
            m = ids_te == t
            vote_d[t] = float(p[m].mean())
            score_d[t] = float(s[m].mean())
    vote = np.array([vote_d[t] for t in trials])
    score = np.array([score_d[t] for t in trials])
    pred = np.array([int(round(v)) for v in vote])
    return pred, vote, score


def main():
    windows_df, features = get_features()
    ids = windows_df["trial_id"].values.astype(str)
    y_win = windows_df["binary_label"].values.astype(int)

    tbl = build_table(windows_df)
    trials = tbl["trial_id"].values
    y = tbl["y"].values.astype(int)
    win_groups = (
        pd.Series(tbl["experiment"].values, index=tbl["trial_id"]).reindex(ids).values
    )

    print("\n=== EXPERIMENT STRUCTURE ===")
    sizes = tbl.groupby("experiment").size()
    print(f"recordings: {len(tbl)}, distinct experiments: {len(sizes)}")
    print("recordings per experiment:", sizes.value_counts().to_dict())
    print("(expected {2: 40} if every experiment has an A1 and an A2 recording)")

    pred, vote, score = loto_grouped(features, y_win, ids, win_groups, trials, CLASS_WEIGHT)
    n, n0, n1 = len(y), int((y == 0).sum()), int((y == 1).sum())
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    print("\n=== LEAVE-ONE-EXPERIMENT-OUT RESULTS ===")
    print(f"class_weight={CLASS_WEIGHT}; recordings: No-Leak={n0}, Leak={n1}; "
          f"baseline={max(n0, n1) / n:.3f}")
    lo, hi = wilson(int((pred == y).sum()), n)
    print(f"accuracy:          {(pred == y).mean():.3f}  (naive 95% CI {lo:.3f}-{hi:.3f}; "
          f"too narrow, effective n is about the number of experiments)")
    print(f"balanced accuracy: {balanced_accuracy_score(y, pred):.3f}")
    print(f"ROC-AUC:           {roc_auc_score(y, score):.3f}")
    print(f"No-Leak recall:    {recall_score(y, pred, pos_label=0):.3f}  ({tn}/{n0})")
    print(f"Leak recall:       {recall_score(y, pred, pos_label=1):.3f}  ({tp}/{n1})")
    print(pd.DataFrame([[tn, fp], [fn, tp]],
                       index=["true No-Leak", "true Leak"],
                       columns=["pred No-Leak", "pred Leak"]))
    print("\nFor comparison, leave-one-recording-out gave: accuracy 0.812, "
          "balanced accuracy 0.742, No-Leak recall 0.625, Leak recall 0.859")

    tmp = tbl.copy()
    tmp["pred"] = pred
    tmp["vote_leak"] = vote
    tmp["correct"] = (tmp["y"] == tmp["pred"]).astype(int)
    print("\n-- by leak_type --")
    print(tmp.groupby("leak_type").agg(
        n=("y", "size"), true_leak=("y", "sum"), predicted_leak=("pred", "sum"),
        accuracy=("correct", "mean"), mean_vote_leak=("vote_leak", "mean"),
    ).round(3))

    print(f"\n=== PERMUTATION TEST (experiment-level shuffles, {N_PERM}) ===")
    exp_y = tbl.groupby("experiment")["y"].first()
    rng = np.random.default_rng(SEED)
    obs = balanced_accuracy_score(y, pred)
    null = []
    t0 = time.time()
    for i in range(N_PERM):
        shuffled = pd.Series(rng.permutation(exp_y.values), index=exp_y.index)
        yp_trial = tbl["experiment"].map(shuffled).values.astype(int)
        yp_win = pd.Series(yp_trial, index=trials).reindex(ids).values.astype(int)
        pp, _, _ = loto_grouped(features, yp_win, ids, win_groups, trials, CLASS_WEIGHT)
        null.append(balanced_accuracy_score(yp_trial, pp))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{N_PERM} done ({time.time() - t0:.0f}s)")
    null = np.array(null)
    p_val = (1 + np.sum(null >= obs)) / (N_PERM + 1)
    print(f"observed balanced accuracy={obs:.3f}, null mean={null.mean():.3f}, "
          f"null 95th pct={np.percentile(null, 95):.3f}, p={p_val:.4f}")

    out = tbl.copy()
    out["y_pred"] = pred
    out["vote_leak_fraction"] = vote
    out["mean_score"] = score
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()