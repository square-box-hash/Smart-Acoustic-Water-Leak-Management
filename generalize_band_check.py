"""
generalize_band_check.py

Tests whether the 1700-2000 Hz band-energy increase (found consistently
across all 4 leak types in one specific network/channel/flow-rate
combination) holds up across ALL combinations of:
  - network layout (Branched, Looped)
  - channel (A1, A2)
  - flow state (0.18 LPS, 0.47 LPS)

For each combination, computes the % change in 1700-2000 Hz band energy
for each leak type vs. that combination's own No Leak baseline, then
summarizes whether the direction (increase) is consistent everywhere,
or only in some configurations.

Run: python generalize_band_check.py
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

SENSOR = "Accelerometer"
LEAK_TYPES = ["Orifice Leak", "Gasket Leak", "Longitudinal Crack", "Circumferential Crack"]
NETWORKS = ["Branched", "Looped"]
CHANNELS = ["A1", "A2"]
STATES = ["0.18 LPS", "0.47 LPS"]

LOWCUT_HZ = 100
HIGHCUT_HZ = 3500
BAND_LOW = 1700
BAND_HIGH = 2000


def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyquist = fs / 2
    sos = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def band_energy(signal, fs, low, high, lowcut, highcut):
    filtered = bandpass_filter(signal, fs, lowcut, highcut)
    n = len(filtered)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    mag = np.abs(np.fft.rfft(filtered)) / n
    mask = (freqs >= low) & (freqs <= high)
    return np.sum(mag[mask] ** 2)


def get_trial(df, sensor, network, channel, leak_type, state):
    match = df[
        (df["sensor"] == sensor)
        & (df["network"] == network)
        & (df["channel"] == channel)
        & (df["leak_type"] == leak_type)
        & (df["state"] == state)
    ]
    if len(match) == 0:
        return None
    return match.iloc[0]


def main():
    df = pd.read_pickle("trials.pkl")

    results = []

    for network in NETWORKS:
        for channel in CHANNELS:
            for state in STATES:
                baseline_trial = get_trial(df, SENSOR, network, channel, "No Leak", state)
                if baseline_trial is None:
                    print(f"MISSING baseline: {network}/{channel}/{state} -- skipping this combo")
                    continue

                baseline_energy = band_energy(
                    np.asarray(baseline_trial["signal"], dtype=float),
                    baseline_trial["sample_rate_hz"],
                    BAND_LOW, BAND_HIGH, LOWCUT_HZ, HIGHCUT_HZ,
                )

                for leak_type in LEAK_TYPES:
                    leak_trial = get_trial(df, SENSOR, network, channel, leak_type, state)
                    if leak_trial is None:
                        print(f"MISSING: {network}/{channel}/{state}/{leak_type} -- skipping")
                        continue

                    leak_energy = band_energy(
                        np.asarray(leak_trial["signal"], dtype=float),
                        leak_trial["sample_rate_hz"],
                        BAND_LOW, BAND_HIGH, LOWCUT_HZ, HIGHCUT_HZ,
                    )

                    pct_change = 100 * (leak_energy - baseline_energy) / baseline_energy
                    results.append({
                        "network": network,
                        "channel": channel,
                        "state": state,
                        "leak_type": leak_type,
                        "baseline_energy": baseline_energy,
                        "leak_energy": leak_energy,
                        "pct_change": pct_change,
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv("band_energy_generalization.csv", index=False)

    print("\n=== Full results ===")
    print(results_df.to_string(index=False))

    print("\n=== Summary: is the increase consistent? ===")
    n_total = len(results_df)
    n_increase = (results_df["pct_change"] > 0).sum()
    print(f"{n_increase} / {n_total} leak trials showed INCREASED 1700-2000Hz energy vs their own No-Leak baseline")

    print("\n=== By leak type ===")
    print(results_df.groupby("leak_type")["pct_change"].agg(["mean", "std", "min", "max",
                                                                lambda x: (x > 0).sum()]))

    print("\nSaved full results to band_energy_generalization.csv")


if __name__ == "__main__":
    main()