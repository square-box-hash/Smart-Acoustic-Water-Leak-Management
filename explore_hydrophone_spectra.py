"""
explore_hydrophone_spectra.py

Before assuming time-domain features (mean/std/skew/kurtosis/RMS) are
the wrong fit for Hydrophone, or jumping to a new frequency-domain
feature set, actually LOOK at the Hydrophone spectra first -- same
approach used at the very start of this project for Accelerometer.

Plots No Leak vs each leak type's FFT spectrum for Hydrophone, split by
noise condition (N/NN) since that's a real, confirmed variable for this
sensor specifically (see anomaly_pipeline.py notes).

Run: python explore_hydrophone_spectra.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

NETWORK = "Branched"
CHANNEL = "H1"
LEAK_TYPES = ["Orifice Leak", "Gasket Leak", "Longitudinal Crack", "Circumferential Crack"]

# Hydrophone Nyquist is 4000 Hz (8kHz sample rate). Aquarian H2c spec
# range is 1 Hz - 100 kHz per the dataset paper, so the sensor itself
# isn't the limiting factor here -- the Nyquist limit from the 8kHz
# sampling is. Using a wide, near-full-range filter so we don't
# pre-judge where the interesting content is before looking.
LOWCUT_HZ = 5
HIGHCUT_HZ = 3900


def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyquist = fs / 2
    sos = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def compute_fft(signal, fs):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    mag = np.abs(np.fft.rfft(signal)) / n
    return freqs, mag


def get_trial(df, network, channel, leak_type, noise_flag=None):
    mask = (
        (df["sensor"] == "Hydrophone")
        & (df["network"] == network)
        & (df["channel"] == channel)
        & (df["leak_type"] == leak_type)
    )
    if noise_flag is not None:
        mask &= (df["extra_flag"] == noise_flag)
    match = df[mask]
    if len(match) == 0:
        return None
    return match.iloc[0]


def get_spectrum(trial):
    fs = trial["sample_rate_hz"]
    signal = np.asarray(trial["signal"], dtype=float)
    filtered = bandpass_filter(signal, fs, LOWCUT_HZ, HIGHCUT_HZ)
    return compute_fft(filtered, fs)


def main():
    df = pd.read_pickle("trials.pkl")
    df = df[df["leak_type"].notna()].copy()

    # Prefer trials WITH background noise (N) since that's the more
    # common/realistic condition and what most of the leak trials use;
    # fall back to whatever's available if not found.
    baseline = get_trial(df, NETWORK, CHANNEL, "No Leak", noise_flag="N")
    if baseline is None:
        baseline = get_trial(df, NETWORK, CHANNEL, "No Leak")
    if baseline is None:
        print("No Hydrophone No-Leak trial found for this network/channel -- adjust config.")
        return

    freqs_baseline, mag_baseline = get_spectrum(baseline)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    fig.suptitle(
        f"Hydrophone | {NETWORK} | {CHANNEL} -- No Leak vs each Leak Type",
        fontsize=13,
    )

    for ax, leak_type in zip(axes.flat, LEAK_TYPES):
        leak_trial = get_trial(df, NETWORK, CHANNEL, leak_type, noise_flag=baseline["extra_flag"])
        if leak_trial is None:
            leak_trial = get_trial(df, NETWORK, CHANNEL, leak_type)
        if leak_trial is None:
            ax.set_title(f"{leak_type} (no matching trial found)")
            continue

        freqs_leak, mag_leak = get_spectrum(leak_trial)
        ax.plot(freqs_baseline, mag_baseline, label="No Leak", color="tab:blue", linewidth=0.6, alpha=0.8)
        ax.plot(freqs_leak, mag_leak, label=leak_type, color="tab:red", linewidth=0.6, alpha=0.8)
        ax.set_title(leak_type)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.legend(fontsize=8)
        ax.set_xlim(0, HIGHCUT_HZ)

    plt.tight_layout()
    out_path = "hydrophone_spectrum_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()

    # Also zoom in on the low end (0-500 Hz), since that's the region
    # the literature (both papers reviewed) flags as most likely to
    # carry leak-relevant information.
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    fig2.suptitle(
        f"Hydrophone | {NETWORK} | {CHANNEL} -- Zoomed to 0-500 Hz",
        fontsize=13,
    )
    for ax, leak_type in zip(axes2.flat, LEAK_TYPES):
        leak_trial = get_trial(df, NETWORK, CHANNEL, leak_type, noise_flag=baseline["extra_flag"])
        if leak_trial is None:
            leak_trial = get_trial(df, NETWORK, CHANNEL, leak_type)
        if leak_trial is None:
            continue
        freqs_leak, mag_leak = get_spectrum(leak_trial)
        ax.plot(freqs_baseline, mag_baseline, label="No Leak", color="tab:blue", linewidth=0.8, alpha=0.8)
        ax.plot(freqs_leak, mag_leak, label=leak_type, color="tab:red", linewidth=0.8, alpha=0.8)
        ax.set_title(leak_type)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 500)

    plt.tight_layout()
    out_path2 = "hydrophone_spectrum_zoomed.png"
    plt.savefig(out_path2, dpi=150)
    print(f"Saved plot to {out_path2}")
    plt.show()


if __name__ == "__main__":
    main()