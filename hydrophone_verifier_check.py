"""
hydrophone_verifier_check.py

Tests whether the near-DC (0-20 Hz) energy drop spotted visually in
explore_hydrophone_spectra.py is a CONSISTENT pattern across all
Hydrophone trials -- not just the one network/channel combo we happened
to plot. This determines whether it's usable as a lightweight
"verifier" signal (does the expected drop occur, yes/no) to sanity-check
a primary classifier's leak call, rather than a coincidence from one
example (the same mistake we caught ourselves making earlier with the
Accelerometer 1700-2000 Hz band).

For every Hydrophone leak trial, compares its 0-20 Hz band energy
against its own matching No-Leak baseline (same network/channel/state/
noise condition where possible) and checks whether it actually dropped.

Run: python hydrophone_verifier_check.py
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

BAND_LOW = 0.5   # avoid true DC (0 Hz) which can be a filter/offset artifact
BAND_HIGH = 20

LOWCUT_HZ = 0.5
HIGHCUT_HZ = 3900


def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyquist = fs / 2
    sos = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def band_energy(signal, fs, low, high):
    filtered = bandpass_filter(signal, fs, LOWCUT_HZ, HIGHCUT_HZ)
    n = len(filtered)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    mag = np.abs(np.fft.rfft(filtered)) / n
    mask = (freqs >= low) & (freqs <= high)
    return np.sum(mag[mask] ** 2)


def main():
    df = pd.read_pickle("trials.pkl")
    df = df[(df["sensor"] == "Hydrophone") & (df["leak_type"].notna())].copy()

    results = []
    for _, leak_row in df[df["leak_type"] != "No Leak"].iterrows():
        # Try to find the closest matching No-Leak baseline: same
        # network/channel/state, and same noise flag if possible.
        candidates = df[
            (df["leak_type"] == "No Leak")
            & (df["network"] == leak_row["network"])
            & (df["channel"] == leak_row["channel"])
            & (df["state"] == leak_row["state"])
        ]
        if len(candidates) == 0:
            continue

        exact_noise_match = candidates[candidates["extra_flag"] == leak_row["extra_flag"]]
        baseline_row = exact_noise_match.iloc[0] if len(exact_noise_match) > 0 else candidates.iloc[0]

        leak_energy = band_energy(
            np.asarray(leak_row["signal"], dtype=float), leak_row["sample_rate_hz"],
            BAND_LOW, BAND_HIGH,
        )
        baseline_energy = band_energy(
            np.asarray(baseline_row["signal"], dtype=float), baseline_row["sample_rate_hz"],
            BAND_LOW, BAND_HIGH,
        )

        dropped = leak_energy < baseline_energy
        pct_change = 100 * (leak_energy - baseline_energy) / baseline_energy if baseline_energy > 0 else np.nan

        results.append({
            "network": leak_row["network"],
            "channel": leak_row["channel"],
            "state": leak_row["state"],
            "leak_type": leak_row["leak_type"],
            "noise_flag": leak_row["extra_flag"],
            "baseline_noise_flag": baseline_row["extra_flag"],
            "leak_energy": leak_energy,
            "baseline_energy": baseline_energy,
            "pct_change": pct_change,
            "dropped": dropped,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("hydrophone_verifier_check.csv", index=False)

    print(f"Total leak trials checked: {len(results_df)}")
    n_dropped = results_df["dropped"].sum()
    print(f"Dropped as expected: {n_dropped} / {len(results_df)} "
          f"({100 * n_dropped / len(results_df):.1f}%)")

    print("\n=== By leak type ===")
    print(results_df.groupby("leak_type")["dropped"].agg(["sum", "count", "mean"]))

    print("\n=== By network ===")
    print(results_df.groupby("network")["dropped"].agg(["sum", "count", "mean"]))

    print("\n=== Cases where noise condition didn't match exactly (less reliable comparison) ===")
    mismatched = results_df[results_df["noise_flag"] != results_df["baseline_noise_flag"]]
    print(f"{len(mismatched)} / {len(results_df)} comparisons used a mismatched noise baseline")

    print("\nFull results saved to hydrophone_verifier_check.csv")


if __name__ == "__main__":
    main()