"""
Tiny artefact / leakage check for the frozen CWT scale-power + SVM pipeline.

Part 1: feature ablations (leave-one-experiment-out, same settings as before)
    - Where does the signal live? loudness only? spectral shape? which band?
Part 2: cross-network transfer
    - Train on one network (Looped / Branched), test on the other.

Put in the same folder as grouped_check.py and full_results_wst.py.
Run:  python tiny_leakage_test.py     (takes about a minute)
"""
import numpy as np
import pandas as pd
import pywt
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from full_results_wst import CLASS_WEIGHT, get_features
from grouped_check import build_table, loto_grouped
from wst_svm_binary_loto import EFFECTIVE_SAMPLE_RATE, RANDOM_STATE, WINDOW_LEN


def summarize(name, y, pred):
    return {
        "test": name,
        "accuracy": round(float((pred == y).mean()), 3),
        "balanced_acc": round(float(balanced_accuracy_score(y, pred)), 3),
        "recall_NoLeak": round(float(recall_score(y, pred, pos_label=0, zero_division=0)), 3),
        "recall_Leak": round(float(recall_score(y, pred, pos_label=1, zero_division=0)), 3),
    }


def train_test(feats, y_win, ids, train_mask, test_mask, weights):
    sc = StandardScaler()
    Xtr = sc.fit_transform(feats[train_mask])
    Xte = sc.transform(feats[test_mask])
    clf = SVC(kernel="rbf", class_weight=weights, random_state=RANDOM_STATE)
    clf.fit(Xtr, y_win[train_mask])
    p = clf.predict(Xte)
    ids_te = ids[test_mask]
    return {t: int(round(float(p[ids_te == t].mean()))) for t in np.unique(ids_te)}


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

    scales = np.geomspace(2, WINDOW_LEN / 8, num=features.shape[1])
    freqs = pywt.scale2frequency("morl", scales) * EFFECTIVE_SAMPLE_RATE
    print("Approx. centre frequency of each feature (Hz):")
    print(np.round(freqs, 1))

    eps = 1e-12
    total = features.sum(axis=1, keepdims=True)
    variants = {
        "all features (reference)": features,
        "total energy only (1 feature)": np.log10(total + eps),
        "spectral shape only (level removed)": np.log10(features / (total + eps) + eps),
        "below 20 Hz only": features[:, freqs < 20],
        "20-200 Hz only": features[:, (freqs >= 20) & (freqs < 200)],
        "above 200 Hz only": features[:, freqs >= 200],
        "all except below 20 Hz": features[:, freqs >= 20],
    }

    print("\n=== PART 1: FEATURE ABLATIONS (leave-one-experiment-out) ===")
    rows = []
    for name, feats in variants.items():
        if feats.shape[1] == 0:
            continue
        pred, _, _ = loto_grouped(feats, y_win, ids, win_groups, trials, CLASS_WEIGHT)
        r = summarize(name, y, pred)
        r["n_features"] = feats.shape[1]
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== PART 2: CROSS-NETWORK TRANSFER ===")
    net_win = pd.Series(tbl["network"].values, index=tbl["trial_id"]).reindex(ids).values
    y_by_trial = tbl.set_index("trial_id")["y"]
    rows = []
    for a in sorted(set(net_win)):
        for b in sorted(set(net_win)):
            if a == b:
                continue
            out = train_test(features, y_win, ids, net_win == a, net_win == b, CLASS_WEIGHT)
            t_ids = list(out.keys())
            yt = y_by_trial.loc[t_ids].values.astype(int)
            pr = np.array([out[t] for t in t_ids])
            r = summarize(f"train {a} -> test {b}", yt, pr)
            r["n_test_recordings"] = len(t_ids)
            rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nReference: leave-one-experiment-out balanced accuracy was 0.734; "
          "chance for balanced accuracy is about 0.5.")


if __name__ == "__main__":
    main()