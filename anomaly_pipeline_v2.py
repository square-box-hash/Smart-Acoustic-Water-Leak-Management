"""
anomaly_pipeline_v2.py

Two changes from anomaly_pipeline.py:

1. WINDOW OVERLAP increased from 50% to 90%, matching common practice
   in the literature (e.g. FiT-WST+ uses 90% overlap). This produces
   many more windows per trial, giving trial-level majority voting more
   votes to work with (previously averaging only 2.2 windows/trial in
   the test set -- too few for voting to meaningfully smooth errors).

2. TRANSIENT-AWARE WINDOWING: Transient trials have an abrupt flow
   change at approximately second 20 (per the dataset's source paper --
   flow drops from 0.47 L/s to 0 L/s). A window that straddles this
   transition mixes two different physical flow states into one
   feature vector, which likely explains why Transient trials were
   disproportionately misclassified in the previous run (8 of 11
   misclassified trials involved Transient or a Longitudinal-Crack
   confusion). For Transient trials specifically, windows within a
   buffer zone around second 20 are excluded, so every window comes
   from a single, stable flow regime (either the 0.47 L/s phase before,
   or the 0 L/s phase after).

Run: python anomaly_pipeline_v2.py
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import skew, kurtosis

FILTER_RANGES = {
    "Accelerometer": (0.5, 3000),
    "DynamicPressureSensor": (10, 20000),
    "Hydrophone": (10, 3900),
}

WINDOW_SECONDS = 3.0
OVERLAP = 0.90  # increased from 0.50

# Transient flow change happens at ~20s into the 30s recording (per
# Aghashahi et al. 2023). Exclude windows whose time range falls within
# this buffer around the transition, so no window straddles it.
TRANSIENT_CHANGE_SECOND = 20.0
TRANSIENT_BUFFER_SECONDS = 1.5  # exclude windows overlapping [18.5, 21.5]

EPSILON = 1e-12


def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyquist = fs / 2
    highcut = min(highcut, nyquist * 0.99)
    sos = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def make_windows(signal, fs, window_seconds, overlap, is_transient=False):
    window_size = int(window_seconds * fs)
    step = max(1, int(window_size * (1 - overlap)))
    if window_size >= len(signal):
        return [(signal, 0.0)]

    windows = []
    start = 0
    while start + window_size <= len(signal):
        start_time = start / fs
        end_time = (start + window_size) / fs

        if is_transient:
            buffer_start = TRANSIENT_CHANGE_SECOND - TRANSIENT_BUFFER_SECONDS
            buffer_end = TRANSIENT_CHANGE_SECOND + TRANSIENT_BUFFER_SECONDS
            # skip this window if it overlaps the exclusion buffer at all
            if not (end_time <= buffer_start or start_time >= buffer_end):
                start += step
                continue

        windows.append((signal[start:start + window_size], start_time))
        start += step

    return windows if windows else [(signal, 0.0)]  # fallback: keep at least one window


def extract_window_features(window, fs):
    n = len(window)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    mag = np.abs(np.fft.rfft(window)) / n
    dominant_frequency = freqs[np.argmax(mag)] if len(mag) else np.nan

    return {
        "mean": np.mean(window),
        "std": np.std(window),
        "skewness": skew(window),
        "kurtosis": kurtosis(window),
        "rms": np.sqrt(np.mean(window ** 2)),
        "dominant_frequency": dominant_frequency,
    }


FEATURE_COLS = ["mean", "std", "skewness", "kurtosis", "rms", "dominant_frequency"]


def build_windowed_feature_table(df):
    rows = []
    df = df[df["leak_type"].notna()].copy()

    for idx, row in df.iterrows():
        sensor = row["sensor"]
        fs = row["sample_rate_hz"]
        lowcut, highcut = FILTER_RANGES.get(sensor, (10, fs / 2 * 0.9))
        signal = np.asarray(row["signal"], dtype=float)
        is_transient = (row["state"] == "Transient")

        try:
            filtered = bandpass_filter(signal, fs, lowcut, highcut)
        except Exception as e:
            print(f"Filter failed for a {sensor} trial ({row.get('raw_name', '?')}): {e}")
            continue

        windows = make_windows(filtered, fs, WINDOW_SECONDS, OVERLAP, is_transient=is_transient)
        for w_idx, (window, start_time) in enumerate(windows):
            feats = extract_window_features(window, fs)
            rows.append({
                "sensor": sensor,
                "network": row["network"],
                "channel": row["channel"],
                "leak_type": row["leak_type"],
                "state": row["state"],
                "noise_flag": row.get("extra_flag", None),
                "window_idx": w_idx,
                "window_start_time": start_time,
                "trial_file": row.get("raw_name", row.get("file_path", "?")),
                **feats,
            })
    return pd.DataFrame(rows)


def main():
    df = pd.read_pickle("trials.pkl")

    print("Building windowed feature table (90% overlap, transient-aware)...")
    windowed = build_windowed_feature_table(df)
    print(f"Generated {len(windowed)} windows from {len(df)} trials.")
    print(windowed["sensor"].value_counts())

    print("\nWindows per trial (should be noticeably higher than v1's 50% overlap run):")
    print(windowed.groupby("trial_file").size().describe())

    print("\nWindows per trial, Transient vs other states (Accelerometer only, for a quick check):")
    accel = windowed[windowed["sensor"] == "Accelerometer"]
    print(accel.groupby(accel["state"] == "Transient").size())

    windowed.to_pickle("windowed_v2.pkl")
    windowed.to_csv("windowed_v2.csv", index=False)
    print("\nSaved to windowed_v2.pkl / .csv")


if __name__ == "__main__":
    main()