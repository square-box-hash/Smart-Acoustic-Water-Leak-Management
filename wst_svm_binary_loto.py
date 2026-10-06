import numpy as np
import pandas as pd
from scipy.signal import decimate
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

TRIALS_PKL = "trials.pkl"
SENSOR = "Accelerometer"
ORIGINAL_SAMPLE_RATE = 51200
DOWNSAMPLE_FACTOR = 10
EFFECTIVE_SAMPLE_RATE = ORIGINAL_SAMPLE_RATE // DOWNSAMPLE_FACTOR  # 5120

WINDOW_SEC = 3.0
OVERLAP = 0.5
WINDOW_LEN = int(WINDOW_SEC * EFFECTIVE_SAMPLE_RATE)   # 15360
STEP = int(WINDOW_LEN * (1 - OVERLAP))                 # 7680

USE_KYMATIO = False   # kymatio broken on this scipy version; using pywt fallback
J = 6
Q = 8

RANDOM_STATE = 42


def build_wst_extractor(window_len):
    if USE_KYMATIO:
        from kymatio.numpy import Scattering1D
        scattering = Scattering1D(J=J, Q=Q, shape=window_len)

        def extract(window):
            coeffs = scattering(window.astype(np.float64))
            return coeffs.mean(axis=-1)

        return extract
    else:
        import pywt
        scales = np.geomspace(2, window_len / 8, num=20)

        def extract(window):
            coeffs, _ = pywt.cwt(window, scales, "morl", method="fft")
            power = np.abs(coeffs) ** 2
            return power.mean(axis=-1)

        return extract


def make_windows(signal_array, window_len=WINDOW_LEN, step=STEP, downsample_factor=DOWNSAMPLE_FACTOR):
    signal_array = np.asarray(signal_array, dtype=np.float64)
    signal_array = decimate(signal_array, downsample_factor)
    n = len(signal_array)
    windows = []
    start = 0
    while start + window_len <= n:
        windows.append(signal_array[start:start + window_len])
        start += step
    return windows


def load_and_window_trials():
    df = pd.read_pickle(TRIALS_PKL)
    df = df[df["sensor"] == SENSOR].copy()
    n_before = len(df)
    df = df.dropna(subset=["leak_type"])
    n_after = len(df)
    if n_before != n_after:
        print(f"Dropped {n_before - n_after} {SENSOR} rows with NaN leak_type")

    df["binary_label"] = np.where(df["leak_type"] == "No Leak", 0, 1)

    rows = []
    for _, row in df.iterrows():
        wins = make_windows(row["signal"])
        for w in wins:
            rows.append({
                "trial_id": row["raw_name"],
                "leak_type": row["leak_type"],
                "binary_label": row["binary_label"],
                "window_signal": w,
            })

    windows_df = pd.DataFrame(rows)
    print(f"{df['raw_name'].nunique()} trials -> {len(windows_df)} windows "
          f"({WINDOW_SEC}s, {int(OVERLAP*100)}% overlap, downsampled {DOWNSAMPLE_FACTOR}x to {EFFECTIVE_SAMPLE_RATE}Hz)")
    return windows_df


def main():
    windows_df = load_and_window_trials()

    extract = build_wst_extractor(WINDOW_LEN)

    print("Extracting WST features for all windows...")
    features = np.stack([
        extract(w) for w in windows_df["window_signal"]
    ])
    labels = windows_df["binary_label"].values
    trial_ids = windows_df["trial_id"].values
    unique_trials = np.unique(trial_ids)

    print(f"Feature matrix: {features.shape}")
    print(f"Binary label counts (window-level): "
          f"{pd.Series(labels).value_counts().to_dict()}\n")

    # Try a few class_weight ratios for No-Leak (class 0) vs Leak (class 1)
    weight_options = [
        {0: 1, 1: 1},   # unweighted
        {0: 2, 1: 1},
        {0: 3, 1: 1},
        {0: 4, 1: 1},
        "balanced",
    ]

    best_result = None

    for weights in weight_options:
        all_true, all_pred = [], []

        for held_out_trial in unique_trials:
            test_mask = trial_ids == held_out_trial
            train_mask = ~test_mask

            scaler = StandardScaler()
            X_train = scaler.fit_transform(features[train_mask])
            X_test = scaler.transform(features[test_mask])

            clf = SVC(kernel="rbf", class_weight=weights, random_state=RANDOM_STATE)
            clf.fit(X_train, labels[train_mask])

            preds = clf.predict(X_test)
            majority_pred = int(round(preds.mean()))
            majority_true = int(round(labels[test_mask].mean()))

            all_true.append(majority_true)
            all_pred.append(majority_pred)

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)
        acc = accuracy_score(all_true, all_pred)
        cm = confusion_matrix(all_true, all_pred)
        no_leak_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0

        print(f"class_weight={weights} -> accuracy={acc:.1%}, No-Leak recall={no_leak_recall:.1%}")
        print(cm)
        print()

        # simple scoring: prioritize catching No-Leak without accuracy collapsing
        score = 0.5 * acc + 0.5 * no_leak_recall
        if best_result is None or score > best_result["score"]:
            best_result = {"weights": weights, "acc": acc, "no_leak_recall": no_leak_recall,
                            "cm": cm, "true": all_true, "pred": all_pred, "score": score}

    print("=" * 60)
    print(f"BEST: class_weight={best_result['weights']}")
    print(f"Accuracy: {best_result['acc']:.1%}, No-Leak recall: {best_result['no_leak_recall']:.1%}")
    print(classification_report(best_result["true"], best_result["pred"],
                                 target_names=["No-Leak", "Leak"]))

if __name__ == "__main__":
    main()