"""
Joint (gasket) leak analysis with the FROZEN pipeline.

Same features (Morlet CWT scale-power, 3 s windows, 50% overlap, 5120 Hz), same RBF SVM,
same leave-one-experiment-out split (both channels of an experiment held out together).
Only the class weights differ: "balanced" (a fixed rule, NOT tuned), because the class
ratio here is not the 80/20 of the Leak vs No-Leak task.

Task A: Gasket Leak vs No Leak
Task B: Gasket Leak vs cut-type leaks (Orifice, Circumferential, Longitudinal)

Write your expectation in the logbook BEFORE running.
Needs full_results_wst.py and grouped_check.py in the same folder.
Run:  python joint_tasks.py
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

from full_results_wst import get_features
from grouped_check import build_table, loto_grouped

# ---------------- CONFIG ----------------
GASKET = ["Gasket Leak"]
NO_LEAK = ["No Leak"]
CUTS = ["Circumferential Crack", "Longitudinal Crack", "Orifice Leak"]
WEIGHTS = "balanced"
N_PERM = 100
SEED = 0
# ----------------------------------------


def run_task(title, tbl, ids, features, pos_names, neg_names, pos_label, neg_label):
    sub = tbl[tbl["leak_type"].isin(pos_names + neg_names)].copy()
    sub["y"] = sub["leak_type"].isin(pos_names).astype(int)
    trials = sub["trial_id"].values
    y = sub["y"].values

    win_mask = np.isin(ids, trials)
    feats = features[win_mask]
    ids_s = ids[win_mask]
    groups_s = pd.Series(sub["experiment"].values, index=trials).reindex(ids_s).values
    y_win = pd.Series(y, index=trials).reindex(ids_s).values.astype(int)

    pred, vote, score = loto_grouped(feats, y_win, ids_s, groups_s, trials, WEIGHTS)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    n_exp = sub["experiment"].nunique()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    print(f"\n=== {title} ===")
    print(f"class_weight={WEIGHTS}; leave-one-experiment-out")
    print(f"recordings: {neg_label}={n0}, {pos_label}={n1}; experiments: {n_exp}")
    print(f"majority-class baseline accuracy: {max(n0, n1) / len(y):.3f}")
    print(f"accuracy:          {(pred == y).mean():.3f}")
    print(f"balanced accuracy: {balanced_accuracy_score(y, pred):.3f}")
    print(f"ROC-AUC:           {roc_auc_score(y, score):.3f}")
    print(f"recall {neg_label}: {recall_score(y, pred, pos_label=0):.3f}  ({tn}/{n0})")
    print(f"recall {pos_label}: {recall_score(y, pred, pos_label=1):.3f}  ({tp}/{n1})")
    print(pd.DataFrame([[tn, fp], [fn, tp]],
                       index=[f"true {neg_label}", f"true {pos_label}"],
                       columns=[f"pred {neg_label}", f"pred {pos_label}"]))

    tmp = sub.copy()
    tmp["pred"] = pred
    tmp["vote_pos"] = vote
    print(f"\n-- share predicted as {pos_label}, by leak_type --")
    print(tmp.groupby("leak_type").agg(
        n=("y", "size"),
        predicted_as_pos=("pred", "mean"),
        mean_vote_pos=("vote_pos", "mean"),
    ).round(3))

    print(f"\n-- permutation test ({N_PERM} experiment-level label shuffles) --")
    exp_y = sub.groupby("experiment")["y"].first()
    rng = np.random.default_rng(SEED)
    obs = balanced_accuracy_score(y, pred)
    null = []
    t0 = time.time()
    for i in range(N_PERM):
        sh = pd.Series(rng.permutation(exp_y.values), index=exp_y.index)
        yp = sub["experiment"].map(sh).values.astype(int)
        yp_win = pd.Series(yp, index=trials).reindex(ids_s).values.astype(int)
        pp, _, _ = loto_grouped(feats, yp_win, ids_s, groups_s, trials, WEIGHTS)
        null.append(balanced_accuracy_score(yp, pp))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{N_PERM} done ({time.time() - t0:.0f}s)")
    null = np.array(null)
    p_val = (1 + np.sum(null >= obs)) / (N_PERM + 1)
    print(f"observed balanced accuracy={obs:.3f}, null mean={null.mean():.3f}, "
          f"null 95th pct={np.percentile(null, 95):.3f}, p={p_val:.4f}")

    out = sub[["trial_id", "leak_type", "experiment"]].copy()
    out["y_true"] = y
    out["y_pred"] = pred
    out["vote_pos"] = vote
    out["score"] = score
    fname = "joint_taskA.csv" if "A" in title.split(":")[0] else "joint_taskB.csv"
    out.to_csv(fname, index=False)
    print(f"saved {fname}")


def main():
    windows_df, features = get_features()
    ids = windows_df["trial_id"].values.astype(str)
    tbl = build_table(windows_df)

    run_task("Task A: Gasket vs No Leak", tbl, ids, features,
             GASKET, NO_LEAK, "Gasket", "NoLeak")
    run_task("Task B: Gasket vs cut-type leaks", tbl, ids, features,
             GASKET, CUTS, "Gasket", "Cuts")


if __name__ == "__main__":
    main()      