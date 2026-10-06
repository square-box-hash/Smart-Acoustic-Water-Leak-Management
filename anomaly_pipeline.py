"""
anomaly_pipeline.py

Combines two ideas:
  1. Al Ghasheem et al. (2025)'s proven feature set for this exact dataset/
     sensor combo: mean, std, skewness, kurtosis, RMS -- simple time-domain
     statistics, no FFT needed for these five (though we keep dominant
     frequency too, since we already have a working FFT pipeline).
  2. An anomaly-score framing instead of multi-class classification:
     for each sensor type, build a "healthy" reference distribution from
     ONLY the No-Leak trials, then score every trial (leak or not) by its
     Mahalanobis distance from that healthy centroid. This measures HOW
     FAR a sample deviates from normal, accounting for how the healthy
     features naturally co-vary -- not just raw distance.

     Mahalanobis distance is used (not simple z-score / Euclidean
     distance) because it accounts for correlations between features:
     if RMS and variance naturally move together in healthy data, a
     sample that moves them together by a lot is less surprising than
     one that moves only one of them -- Mahalanobis distance captures
     this, Euclidean distance does not.

  Also: features are computed over SLIDING WINDOWS within each 30s trial
  (not one vector per whole trial), following the windowing approach used
  by every strong result in the literature on this dataset -- this turns
  282 trials into many more training/evaluation samples.

  Sensor-dependency reduction: the healthy reference and the anomaly
  score are computed SEPARATELY per sensor type (Accelerometer, Dynamic
  Pressure Sensor, Hydrophone), so raw units/scales never mix -- what
  gets compared across sensors is each one's own "how many typical
  healthy-deviations away" number, not raw amplitude.

Run: python anomaly_pipeline.py
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import mahalanobis

# Sensor-specific filter ranges, refined after reading the dataset's
# source paper (Aghashahi et al. 2023): the accelerometer's physical
# measurement range is 0.5-3000 Hz (PCB 333B50 datasheet), so filtering
# beyond that is filtering into a region the sensor can't reliably
# measure anyway. Dynamic Pressure Sensor has a much higher resonant
# frequency spec (>=500 kHz) so is not capped the same way here -- kept
# wide pending further investigation.
FILTER_RANGES = {
    "Accelerometer": (0.5, 3000),
    "DynamicPressureSensor": (10, 20000),
    "Hydrophone": (10, 3900),  # hydrophone Nyquist is 4000 Hz (8kHz sample rate)
}

WINDOW_SECONDS = 3.0
OVERLAP = 0.5  # 50% overlap between consecutive windows

EPSILON = 1e-12


def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyquist = fs / 2
    highcut = min(highcut, nyquist * 0.99)  # avoid filter design errors near Nyquist
    sos = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def make_windows(signal, fs, window_seconds, overlap):
    window_size = int(window_seconds * fs)
    step = int(window_size * (1 - overlap))
    if window_size >= len(signal):
        return [signal]  # trial shorter than one window -- use whole trial
    windows = []
    start = 0
    while start + window_size <= len(signal):
        windows.append(signal[start:start + window_size])
        start += step
    return windows


def extract_window_features(window, fs):
    """Al Ghasheem-style time-domain features, plus dominant frequency
    from our existing FFT approach (kept since we already validated it)."""
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
    # Drop rows with no usable leak_type label -- these are the
    # Background Noise_H1/H2 hydrophone files, which don't follow the
    # standard T_L_F_S# naming convention and aren't tied to a specific
    # leak/no-leak experimental trial. They're not useful for the
    # healthy-vs-leak comparison here.
    df = df[df["leak_type"].notna()].copy()

    for idx, row in df.iterrows():
        sensor = row["sensor"]
        fs = row["sample_rate_hz"]
        lowcut, highcut = FILTER_RANGES.get(sensor, (10, fs / 2 * 0.9))
        signal = np.asarray(row["signal"], dtype=float)

        try:
            filtered = bandpass_filter(signal, fs, lowcut, highcut)
        except Exception as e:
            print(f"Filter failed for a {sensor} trial ({row.get('raw_name', '?')}): {e}")
            continue

        windows = make_windows(filtered, fs, WINDOW_SECONDS, OVERLAP)
        for w_idx, window in enumerate(windows):
            feats = extract_window_features(window, fs)
            rows.append({
                "sensor": sensor,
                "network": row["network"],
                "channel": row["channel"],
                "leak_type": row["leak_type"],
                "state": row["state"],
                "noise_flag": row.get("extra_flag", None),  # N/NN, hydrophone only
                "window_idx": w_idx,
                "trial_file": row.get("raw_name", row.get("file_path", "?")),
                **feats,
            })
    return pd.DataFrame(rows)


def fit_healthy_reference(healthy_df, feature_cols):
    """Returns (mean_vector, inverse_covariance_matrix) for Mahalanobis
    distance, fit ONLY on No-Leak windows for one sensor."""
    X = healthy_df[feature_cols].dropna().to_numpy()
    mean_vec = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    # regularize slightly to avoid singular matrix issues with small samples
    cov += np.eye(cov.shape[0]) * 1e-8
    inv_cov = np.linalg.inv(cov)
    return mean_vec, inv_cov


def compute_anomaly_scores(df, feature_cols):
    """Computes Mahalanobis anomaly score per window, using a SEPARATE
    healthy reference fit per sensor type -- this is what keeps the
    score sensor-agnostic in its meaning (a "3.0" score means the same
    thing -- roughly 3 typical healthy deviations away -- whether it
    came from an accelerometer or a hydrophone), even though the raw
    features themselves are on totally different physical scales.

    For Hydrophone specifically, background noise (traffic/saw sounds,
    the N/NN flag) varies across trials and plausibly dominates the raw
    signal more than for the pipe-coupled sensors. A single healthy
    reference blending noisy and quiet No-Leak trials would itself be
    an inconsistent, overly-wide "normal" -- so Hydrophone's healthy
    reference (and its scoring) is split by noise_flag instead."""
    scores = np.full(len(df), np.nan)

    for sensor in df["sensor"].unique():
        sensor_mask = df["sensor"] == sensor

        if sensor == "Hydrophone":
            for noise_flag in df.loc[sensor_mask, "noise_flag"].dropna().unique():
                group_mask = sensor_mask & (df["noise_flag"] == noise_flag)
                healthy_mask = group_mask & (df["leak_type"] == "No Leak")
                healthy_df = df[healthy_mask]
                if len(healthy_df) < len(feature_cols) + 1:
                    print(f"Not enough healthy Hydrophone/{noise_flag} windows "
                          f"({len(healthy_df)} available) -- skipping this group.")
                    continue
                mean_vec, inv_cov = fit_healthy_reference(healthy_df, feature_cols)
                for idx, row in df[group_mask].iterrows():
                    x = row[feature_cols].to_numpy(dtype=float)
                    if np.any(np.isnan(x)):
                        continue
                    scores[df.index.get_loc(idx)] = mahalanobis(x, mean_vec, inv_cov)
            continue

        healthy_mask = sensor_mask & (df["leak_type"] == "No Leak")
        healthy_df = df[healthy_mask]
        if len(healthy_df) < len(feature_cols) + 1:
            print(f"Not enough healthy windows for {sensor} to fit a reference "
                  f"({len(healthy_df)} available) -- skipping this sensor.")
            continue

        mean_vec, inv_cov = fit_healthy_reference(healthy_df, feature_cols)

        sensor_df = df[sensor_mask]
        for idx, row in sensor_df.iterrows():
            x = row[feature_cols].to_numpy(dtype=float)
            if np.any(np.isnan(x)):
                continue
            scores[df.index.get_loc(idx)] = mahalanobis(x, mean_vec, inv_cov)

    return scores


def main():
    df = pd.read_pickle("trials.pkl")

    print("Building windowed feature table (this may take a little while)...")
    windowed = build_windowed_feature_table(df)
    print(f"Generated {len(windowed)} windows from {len(df)} trials.")
    print(windowed["sensor"].value_counts())

    print("\nComputing Mahalanobis anomaly scores per sensor (healthy reference = No Leak only)...")
    windowed["anomaly_score"] = compute_anomaly_scores(windowed, FEATURE_COLS)

    windowed.to_pickle("windowed_anomaly_scores.pkl")
    windowed.to_csv("windowed_anomaly_scores.csv", index=False)

    print("\n=== Anomaly score summary by leak type (across all sensors) ===")
    print(windowed.groupby("leak_type")["anomaly_score"].describe())

    print("\n=== Anomaly score summary by sensor + leak type ===")
    print(windowed.groupby(["sensor", "leak_type"])["anomaly_score"].mean().unstack())

    # Simple separation check: does "No Leak" have a lower average score
    # than every leak type, per sensor? This is the core question --
    # does the anomaly framing actually work.
    print("\n=== Separation check: mean anomaly score, No Leak vs each Leak Type ===")
    for sensor in windowed["sensor"].unique():
        sub = windowed[windowed["sensor"] == sensor]
        no_leak_mean = sub[sub["leak_type"] == "No Leak"]["anomaly_score"].mean()
        print(f"\n{sensor} -- No Leak mean score: {no_leak_mean:.3f}")
        for leak_type in sub["leak_type"].unique():
            if leak_type == "No Leak":
                continue
            leak_mean = sub[sub["leak_type"] == leak_type]["anomaly_score"].mean()
            direction = "HIGHER (expected)" if leak_mean > no_leak_mean else "LOWER (unexpected)"
            print(f"  {leak_type}: {leak_mean:.3f}  [{direction}]")

    print("\nSaved full windowed feature + anomaly score table to "
          "windowed_anomaly_scores.pkl / .csv")


if __name__ == "__main__":
    main()