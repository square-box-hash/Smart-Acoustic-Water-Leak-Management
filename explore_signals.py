"""
explore_signals.py

Picks one matched pair of trials -- same sensor, network, channel --
one No Leak and one Orifice Leak, and compares them:
  1. Raw waveform (time domain)
  2. Band-pass filtered waveform
  3. FFT magnitude spectrum (before vs after filtering)

This is a diagnostic/exploration script, not the final pipeline --
the goal is to SEE whether a leak signature is visually distinguishable
before building the automated feature-extraction + classifier stage.

Run: python explore_signals.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

# ---------------------------------------------------------------------------
# Config -- adjust these to compare different trials
# ---------------------------------------------------------------------------

SENSOR = "Accelerometer"
NETWORK = "Branched"
CHANNEL = "A1"
STATE = "0.47 LPS"          # leak severity to compare against No Leak baseline
LEAK_TYPE = "Orifice Leak"

# Band-pass filter range (Hz) -- adjust based on what the literature /
# sensor datasheet suggests for leak-relevant frequencies. Starting with
# a broad placeholder range; this should be revisited once you've looked
# at the raw spectra below.
LOWCUT_HZ = 100
HIGHCUT_HZ = 5000


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def bandpass_filter(signal, fs, lowcut, highcut, order=4):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def compute_fft(signal, fs):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    mag = np.abs(np.fft.rfft(signal)) / n
    return freqs, mag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_trial(df, sensor, network, channel, leak_type, state):
    match = df[
        (df["sensor"] == sensor)
        & (df["network"] == network)
        & (df["channel"] == channel)
        & (df["leak_type"] == leak_type)
        & (df["state"] == state)
    ]
    if len(match) == 0:
        raise ValueError(
            f"No trial found for sensor={sensor}, network={network}, "
            f"channel={channel}, leak_type={leak_type}, state={state}"
        )
    if len(match) > 1:
        print(f"Warning: {len(match)} matches found, using the first.")
    return match.iloc[0]


def main():
    df = pd.read_pickle("trials.pkl")

    no_leak = get_trial(df, SENSOR, NETWORK, CHANNEL, "No Leak", STATE)
    leak = get_trial(df, SENSOR, NETWORK, CHANNEL, LEAK_TYPE, STATE)

    fs = no_leak["sample_rate_hz"]
    assert fs == leak["sample_rate_hz"], "Sample rates don't match -- check trial selection."

    sig_no_leak = np.asarray(no_leak["signal"], dtype=float)
    sig_leak = np.asarray(leak["signal"], dtype=float)

    # Trim both to the same length (shorter of the two) for fair comparison
    n = min(len(sig_no_leak), len(sig_leak))
    sig_no_leak = sig_no_leak[:n]
    sig_leak = sig_leak[:n]
    t = np.arange(n) / fs

    # Filter
    filt_no_leak = bandpass_filter(sig_no_leak, fs, LOWCUT_HZ, HIGHCUT_HZ)
    filt_leak = bandpass_filter(sig_leak, fs, LOWCUT_HZ, HIGHCUT_HZ)

    # FFT (on filtered signals)
    freqs_no_leak, mag_no_leak = compute_fft(filt_no_leak, fs)
    freqs_leak, mag_leak = compute_fft(filt_leak, fs)

    # -----------------------------------------------------------------
    # Plot: 3 rows x 2 cols -- raw, filtered, FFT -- No Leak | Leak
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(
        f"{SENSOR} | {NETWORK} | {CHANNEL} | {STATE}  --  No Leak vs {LEAK_TYPE}",
        fontsize=13,
    )

    # Only plot first 2 seconds of time-domain signal for readability
    plot_samples = min(n, int(2 * fs))

    axes[0, 0].plot(t[:plot_samples], sig_no_leak[:plot_samples], linewidth=0.5)
    axes[0, 0].set_title("Raw signal -- No Leak")
    axes[0, 0].set_ylabel("Amplitude")

    axes[0, 1].plot(t[:plot_samples], sig_leak[:plot_samples], linewidth=0.5, color="tab:red")
    axes[0, 1].set_title(f"Raw signal -- {LEAK_TYPE}")

    axes[1, 0].plot(t[:plot_samples], filt_no_leak[:plot_samples], linewidth=0.5)
    axes[1, 0].set_title(f"Filtered ({LOWCUT_HZ}-{HIGHCUT_HZ} Hz) -- No Leak")
    axes[1, 0].set_ylabel("Amplitude")

    axes[1, 1].plot(t[:plot_samples], filt_leak[:plot_samples], linewidth=0.5, color="tab:red")
    axes[1, 1].set_title(f"Filtered ({LOWCUT_HZ}-{HIGHCUT_HZ} Hz) -- {LEAK_TYPE}")

    axes[2, 0].plot(freqs_no_leak, mag_no_leak, linewidth=0.7)
    axes[2, 0].set_title("FFT spectrum -- No Leak")
    axes[2, 0].set_xlabel("Frequency (Hz)")
    axes[2, 0].set_ylabel("Magnitude")
    axes[2, 0].set_xlim(0, HIGHCUT_HZ * 1.2)

    axes[2, 1].plot(freqs_leak, mag_leak, linewidth=0.7, color="tab:red")
    axes[2, 1].set_title(f"FFT spectrum -- {LEAK_TYPE}")
    axes[2, 1].set_xlabel("Frequency (Hz)")
    axes[2, 1].set_xlim(0, HIGHCUT_HZ * 1.2)

    plt.tight_layout()
    out_path = "no_leak_vs_leak_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()

    # -----------------------------------------------------------------
    # Quick numeric summary -- useful for the logbook / discussion
    # -----------------------------------------------------------------
    print("\n--- Quick comparison ---")
    print(f"No Leak  -- RMS: {np.sqrt(np.mean(filt_no_leak**2)):.6f}, "
          f"peak freq: {freqs_no_leak[np.argmax(mag_no_leak)]:.1f} Hz")
    print(f"{LEAK_TYPE} -- RMS: {np.sqrt(np.mean(filt_leak**2)):.6f}, "
          f"peak freq: {freqs_leak[np.argmax(mag_leak)]:.1f} Hz")


if __name__ == "__main__":
    main()