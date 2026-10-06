"""
within_trial_split_test.py

A THIRD, explicitly different validation scheme, to be reported honestly
alongside (not instead of) the trial-level split result:

  - Trial-level split (improve_accelerometer_model.py, corrected version):
    entire trials held out -- tests "can the model recognize a genuinely
    NEW leak event it has never seen anything from?" This is the strict,
    real generalization test. Result: ~33.7% mean accuracy.

  - Within-trial time-split (THIS script): for every trial, an early
    time-segment is used for training and a later time-segment (held
    out, never touched during training) is used for testing. Tests a
    DIFFERENT, narrower question: "given that the model has already
    seen part of THIS SAME recording, can it correctly classify the
    rest of it?" This is a fair, real question for some use cases
    (e.g. continuous monitoring of an already-flagged anomaly) but is
    NOT a test of generalizing to unseen leak events, since the model
    has partial information about the exact trial being tested.

Both numbers are reported so the difference between them itself becomes
a finding: a large gap indicates the model is relying heavily on
trial-specific quirks (background noise texture, exact leak severity on
that day, etc.) rather than a general, class-level signature.

Run: python within_trial_split_test.py
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
TRAIN_FRACTION = 0.833  # 25s train / 5s test out of a 30s trial (~5:1 ratio)
FILTER_RANGE = (0.5, 3000)  # Accelerometer-specific, per dataset paper
WINDOW_SECONDS = 3.0
OVERLAP = 0.90

FEATURE_COLS = ["mean", "std", "skewness", "kurtosis", "rms", "dominant_frequency"]


def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyquist = fs / 2
    highcut = min(highcut, nyquist * 0.99)
    sos = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def extract_features(window, fs):
    n = len(window)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    mag = np.abs(np.fft.rfft(window)) / n
    dominant_frequency = freqs[np.argmax(mag)] if len(mag) else np.nan
    return {
        "mean": np.mean(window), "std": np.std(window),
        "skewness": skew(window), "kurtosis": kurtosis(window),
        "rms": np.sqrt(np.mean(window ** 2)), "dominant_frequency": dominant_frequency,
    }


def make_windows(signal, fs, window_seconds, overlap):
    window_size = int(window_seconds * fs)
    step = max(1, int(window_size * (1 - overlap)))
    windows = []
    start = 0
    while start + window_size <= len(signal):
        windows.append(signal[start:start + window_size])
        start += step
    return windows


def main():
    df = pd.read_pickle("trials.pkl")
    df = df[(df["sensor"] == "Accelerometer") & (df["leak_type"].notna())].copy()
    lowcut, highcut = FILTER_RANGE

    train_rows, test_rows = [], []

    for _, row in df.iterrows():
        fs = row["sample_rate_hz"]
        signal = np.asarray(row["signal"], dtype=float)
        filtered = bandpass_filter(signal, fs, lowcut, highcut)

        split_point = int(len(filtered) * TRAIN_FRACTION)
        train_signal = filtered[:split_point]
        test_signal = filtered[split_point:]

        for window in make_windows(train_signal, fs, WINDOW_SECONDS, OVERLAP):
            feats = extract_features(window, fs)
            train_rows.append({"leak_type": row["leak_type"], "trial_file": row.get("raw_name", "?"), **feats})

        for window in make_windows(test_signal, fs, WINDOW_SECONDS, OVERLAP):
            feats = extract_features(window, fs)
            test_rows.append({"leak_type": row["leak_type"], "trial_file": row.get("raw_name", "?"), **feats})

    train_df = pd.DataFrame(train_rows).dropna(subset=FEATURE_COLS)
    test_df = pd.DataFrame(test_rows).dropna(subset=FEATURE_COLS)

    print(f"Train windows (first {TRAIN_FRACTION*100:.0f}% of each trial): {len(train_df)}")
    print(f"Test windows (last {(1-TRAIN_FRACTION)*100:.0f}% of each trial): {len(test_df)}")
    print(f"\nNote: test windows come from the SAME {df['leak_type'].notna().sum()} trials "
          f"as training -- every trial contributes to both sets.\n")

    X_train, y_train = train_df[FEATURE_COLS], train_df["leak_type"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["leak_type"]

    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced")
    clf.fit(X_train, y_train)

    window_preds = clf.predict(X_test)
    window_acc = accuracy_score(y_test, window_preds)
    print(f"Window-level accuracy (within-trial split): {window_acc:.3f}")

    # trial-level majority vote within this same scheme
    test_df = test_df.copy()
    test_df["predicted"] = window_preds
    from collections import Counter
    trial_results = []
    for trial_file, group in test_df.groupby("trial_file"):
        actual = group["leak_type"].iloc[0]
        predicted = Counter(group["predicted"]).most_common(1)[0][0]
        trial_results.append({"trial_file": trial_file, "actual": actual, "predicted": predicted})
    trial_df = pd.DataFrame(trial_results)
    trial_acc = accuracy_score(trial_df["actual"], trial_df["predicted"])
    print(f"Trial-level accuracy (majority vote, within-trial split): {trial_acc:.3f}")

    print("\n" + "=" * 60)
    print("COMPARISON -- read the gap, not just the numbers")
    print("=" * 60)
    print(f"Within-trial split (this script)      : {trial_acc:.3f}  <- 'recognize the rest of a partially-seen trial'")
    print(f"Trial-level split (previous script)    : 0.337            <- 'recognize a completely new, unseen trial'")
    print(f"\nA large gap between these two numbers would indicate the model "
          f"leans on trial-specific quirks (this exact recording's noise "
          f"texture, exact leak severity that day, etc.) rather than a "
          f"general, class-level signature that transfers to new events.")


if __name__ == "__main__":
    main()