"""
extract_features.py

Systematic feature extraction across ALL trials in the dataset -- this
replaces manual single-band eyeballing (which we found gives misleading
results, e.g. huge % swings caused by near-zero baselines) with a
proper per-trial feature vector.

Features computed per trial (after band-pass filtering):
  - dominant_frequency   : frequency bin with peak magnitude
  - total_energy         : sum of squared magnitude across the passband
  - rms_amplitude        : root-mean-square of the filtered time-domain signal
  - spectral_entropy     : how "spread out" vs "peaky" the spectrum is
  - signal_variance      : variance of the filtered time-domain signal
  - band_energy_1700_2000: energy specifically in the band flagged earlier

Baseline-relative versions of each feature are then computed by
comparing each leak trial to its OWN matching No-Leak baseline
(same sensor, network, channel, state) -- but using a SAFE method this
time: absolute difference and log-ratio, not raw percent-change, since
percent-change blows up when the baseline is near zero (as we found in
generalize_band_check.py).

Run: python extract_features.py
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.stats import entropy

LOWCUT_HZ = 100
HIGHCUT_HZ = 3500
BAND_LOW = 1700
BAND_HIGH = 2000

# small constant to avoid log(0) / division-by-zero without distorting
# real values -- chosen relative to the smallest energies we saw
# (~1e-10) in generalize_band_check.py
EPSILON = 1e-12


def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyquist = fs / 2
    sos = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def extract_features_single(signal, fs):
    filtered = bandpass_filter(signal, fs, LOWCUT_HZ, HIGHCUT_HZ)

    n = len(filtered)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    mag = np.abs(np.fft.rfft(filtered)) / n

    passband_mask = (freqs >= LOWCUT_HZ) & (freqs <= HIGHCUT_HZ)
    mag_pb = mag[passband_mask]
    freqs_pb = freqs[passband_mask]

    dominant_frequency = freqs_pb[np.argmax(mag_pb)] if len(mag_pb) else np.nan
    total_energy = np.sum(mag_pb ** 2)
    rms_amplitude = np.sqrt(np.mean(filtered ** 2))
    signal_variance = np.var(filtered)

    # spectral entropy: normalize spectrum to a probability distribution,
    # then compute Shannon entropy -- flat/noisy spectrum = high entropy,
    # peaky/tonal spectrum = low entropy
    psd = mag_pb ** 2
    psd_norm = psd / (np.sum(psd) + EPSILON)
    spectral_entropy = entropy(psd_norm + EPSILON)

    band_mask = (freqs >= BAND_LOW) & (freqs <= BAND_HIGH)
    band_energy_1700_2000 = np.sum(mag[band_mask] ** 2)

    return {
        "dominant_frequency": dominant_frequency,
        "total_energy": total_energy,
        "rms_amplitude": rms_amplitude,
        "signal_variance": signal_variance,
        "spectral_entropy": spectral_entropy,
        "band_energy_1700_2000": band_energy_1700_2000,
    }


def main():
    df = pd.read_pickle("trials.pkl")

    print(f"Extracting features for {len(df)} trials...")
    feature_rows = []
    for idx, row in df.iterrows():
        signal = np.asarray(row["signal"], dtype=float)
        feats = extract_features_single(signal, row["sample_rate_hz"])
        feature_rows.append(feats)

    feat_df = pd.DataFrame(feature_rows)
    combined = pd.concat([df.drop(columns=["signal"]), feat_df], axis=1)

    # -----------------------------------------------------------------
    # Baseline-relative features: for every non-"No Leak" trial, find
    # its matching No-Leak baseline (same sensor/network/channel/state)
    # and compute SAFE relative features -- absolute difference and
    # log-ratio, not raw percent-change.
    # -----------------------------------------------------------------
    feature_cols = list(feat_df.columns)
    baseline_lookup = combined[combined["leak_type"] == "No Leak"].set_index(
        ["sensor", "network", "channel", "state"]
    )

    rel_rows = []
    for idx, row in combined.iterrows():
        key = (row["sensor"], row["network"], row["channel"], row["state"])
        if row["leak_type"] == "No Leak" or key not in baseline_lookup.index:
            rel_rows.append({f"{c}_absdiff": np.nan for c in feature_cols} |
                             {f"{c}_logratio": np.nan for c in feature_cols})
            continue

        baseline_row = baseline_lookup.loc[key]
        if isinstance(baseline_row, pd.DataFrame):
            baseline_row = baseline_row.iloc[0]  # if multiple matches, take first

        rel = {}
        for c in feature_cols:
            baseline_val = baseline_row[c]
            trial_val = row[c]
            rel[f"{c}_absdiff"] = trial_val - baseline_val
            # log-ratio: safe for near-zero values, symmetric around 0,
            # meaningful even when baseline_val is tiny
            rel[f"{c}_logratio"] = np.log((abs(trial_val) + EPSILON) / (abs(baseline_val) + EPSILON))
        rel_rows.append(rel)

    rel_df = pd.DataFrame(rel_rows)
    final = pd.concat([combined.reset_index(drop=True), rel_df.reset_index(drop=True)], axis=1)

    final.to_pickle("trials_with_features.pkl")
    final.to_csv("trials_with_features.csv", index=False)

    print(f"\nExtracted {len(feature_cols)} base features + baseline-relative versions.")
    print(f"Saved {len(final)} rows to trials_with_features.pkl / .csv")
    print("\nFeature columns:")
    print([c for c in final.columns if c not in df.columns] )

    print("\nSample of extracted features (first 5 rows):")
    preview_cols = ["sensor", "network", "channel", "leak_type", "state",
                     "dominant_frequency", "total_energy", "rms_amplitude",
                     "band_energy_1700_2000_logratio"]
    print(final[preview_cols].head())


if __name__ == "__main__":
    main()