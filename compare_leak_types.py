"""
compare_leak_types.py

Extends the single-pair comparison from explore_signals.py: instead of
comparing just one leak type against No Leak, this overlays ALL leak
types against the same No Leak baseline (same sensor, network, channel,
flow state) on one FFT plot each, so we can check whether the frequency
signature we spotted (~1700-2000 Hz bump) is specific to Orifice Leak,
or common across leak types (which would make it a much stronger,
more general finding).

Run: python compare_leak_types.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

SENSOR = "Accelerometer"
NETWORK = "Branched"
CHANNEL = "A1"
STATE = "0.47 LPS"

LEAK_TYPES = ["Orifice Leak", "Gasket Leak", "Longitudinal Crack", "Circumferential Crack"]

# Narrowed based on the first exploratory result -- almost no meaningful
# energy was observed above ~3500-4000 Hz in either No Leak or Orifice Leak.
LOWCUT_HZ = 100
HIGHCUT_HZ = 3500


def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyquist = fs / 2
    sos = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def compute_fft(signal, fs):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    mag = np.abs(np.fft.rfft(signal)) / n
    return freqs, mag


def get_trial(df, sensor, network, channel, leak_type, state):
    match = df[
        (df["sensor"] == sensor)
        & (df["network"] == network)
        & (df["channel"] == channel)
        & (df["leak_type"] == leak_type)
        & (df["state"] == state)
    ]
    if len(match) == 0:
        raise ValueError(f"No trial for leak_type={leak_type}, state={state}")
    return match.iloc[0]


def get_spectrum(df, leak_type):
    trial = get_trial(df, SENSOR, NETWORK, CHANNEL, leak_type, STATE)
    fs = trial["sample_rate_hz"]
    signal = np.asarray(trial["signal"], dtype=float)
    filtered = bandpass_filter(signal, fs, LOWCUT_HZ, HIGHCUT_HZ)
    return compute_fft(filtered, fs)


def main():
    df = pd.read_pickle("trials.pkl")

    freqs_baseline, mag_baseline = get_spectrum(df, "No Leak")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    fig.suptitle(
        f"{SENSOR} | {NETWORK} | {CHANNEL} | {STATE} -- No Leak vs each Leak Type",
        fontsize=13,
    )

    for ax, leak_type in zip(axes.flat, LEAK_TYPES):
        freqs_leak, mag_leak = get_spectrum(df, leak_type)
        ax.plot(freqs_baseline, mag_baseline, label="No Leak", color="tab:blue", linewidth=0.7, alpha=0.8)
        ax.plot(freqs_leak, mag_leak, label=leak_type, color="tab:red", linewidth=0.7, alpha=0.8)
        ax.set_title(leak_type)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.legend(fontsize=8)
        ax.set_xlim(0, HIGHCUT_HZ)

    plt.tight_layout()
    out_path = "leak_type_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()

    # Numeric check: energy in the 1700-2000 Hz band specifically,
    # since that's the region flagged in the first exploratory result.
    print("\n--- Band energy check (1700-2000 Hz) ---")
    band_mask = (freqs_baseline >= 1700) & (freqs_baseline <= 2000)
    baseline_energy = np.sum(mag_baseline[band_mask] ** 2)
    print(f"No Leak: {baseline_energy:.8f}")
    for leak_type in LEAK_TYPES:
        freqs_leak, mag_leak = get_spectrum(df, leak_type)
        band_mask_leak = (freqs_leak >= 1700) & (freqs_leak <= 2000)
        leak_energy = np.sum(mag_leak[band_mask_leak] ** 2)
        pct_change = 100 * (leak_energy - baseline_energy) / baseline_energy
        print(f"{leak_type}: {leak_energy:.8f}  ({pct_change:+.1f}% vs No Leak)")


if __name__ == "__main__":
    main()
    