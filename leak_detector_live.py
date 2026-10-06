"""
Live pipe-leak detector using MPU6050 + Arduino Nano.

Workflow:
  1. Calibration Phase 1 (DRY): pipe with no water, 60s -- establishes the
     "quiet" baseline (ambient vibration / electrical noise floor).
  2. Calibration Phase 2 (WET, NO LEAK): pipe with water flowing normally,
     no leak, 60s -- establishes the "normal flow" baseline.
  3. Live monitoring: continuously reads new data, extracts the same
     features used for baseline, and flags a leak if the live reading
     deviates significantly from the WET-NO-LEAK baseline (using a
     Mahalanobis-style distance, matching the approach you already
     validated on the Dynamic Pressure Sensor in your dataset analysis).

Requires: pyserial, numpy, scipy
    pip install pyserial numpy scipy

Usage:
    1. Update SERIAL_PORT below to match your Arduino's COM port
       (check Arduino IDE -> Tools -> Port).
    2. Run the script. Follow the on-screen prompts for each phase.
"""

import time
import numpy as np
import serial
from scipy.fft import rfft, rfftfreq

# ------------------------- CONFIG --------------------------------------
SERIAL_PORT = "COM3"          # <-- CHANGE THIS to your Arduino's port
BAUD_RATE = 115200

CALIBRATION_DURATION_SEC = 60
WINDOW_SEC = 2.0               # feature window length for live analysis
LEAK_THRESHOLD_STD = 3.0       # flag as leak if Mahalanobis distance
                                # exceeds this many "standard deviations"
                                # from the WET-NO-LEAK baseline
# -------------------------------------------------------------------------


def connect_serial():
    print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # allow Arduino to reset after serial connection opens
    ser.reset_input_buffer()
    return ser


def read_samples_for(ser, duration_sec, label=""):
    """Reads ax,ay,az lines from serial for duration_sec seconds."""
    print(f"\nRecording '{label}' for {duration_sec}s...")
    samples = []
    t_end = time.time() + duration_sec
    last_print = time.time()

    while time.time() < t_end:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 3:
            continue
        try:
            ax, ay, az = float(parts[0]), float(parts[1]), float(parts[2])
            samples.append((ax, ay, az))
        except ValueError:
            continue

        if time.time() - last_print > 5:
            remaining = int(t_end - time.time())
            print(f"  ...{remaining}s remaining, {len(samples)} samples so far")
            last_print = time.time()

    samples = np.array(samples)
    print(f"  Collected {len(samples)} samples for '{label}'")
    return samples


def extract_features(samples, sample_rate_hz):
    """
    Extracts a small feature vector from a block of (ax,ay,az) samples:
    per-axis RMS, std, and dominant frequency band energy (low-frequency
    band, matching the <1000-2000Hz leak-signature range from your
    dataset literature review).
    """
    features = []
    for axis in range(3):
        signal = samples[:, axis]
        rms = np.sqrt(np.mean(signal ** 2))
        std = np.std(signal)

        # Frequency-domain: energy in a low band (adjust based on your
        # dataset findings -- using 0-500Hz here as a starting point)
        freqs = rfftfreq(len(signal), d=1.0 / sample_rate_hz)
        spectrum = np.abs(rfft(signal))
        band_mask = (freqs >= 0) & (freqs <= 500)
        band_energy = np.sum(spectrum[band_mask] ** 2)

        features.extend([rms, std, band_energy])

    return np.array(features)


def estimate_sample_rate(ser, test_duration=5):
    """Quick check of actual achieved sample rate from the Arduino."""
    print(f"\nEstimating actual sample rate over {test_duration}s...")
    samples = read_samples_for(ser, test_duration, label="rate check")
    rate = len(samples) / test_duration
    print(f"  Estimated sample rate: {rate:.1f} Hz")
    return rate


def mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    return np.sqrt(diff @ inv_cov @ diff)


def main():
    ser = connect_serial()

    sample_rate = estimate_sample_rate(ser)
    window_samples = int(WINDOW_SEC * sample_rate)

    input("\n>>> Set up pipe with NO WATER. Press Enter to start DRY calibration...")
    dry_samples = read_samples_for(ser, CALIBRATION_DURATION_SEC, label="DRY baseline")

    input("\n>>> Now fill pipe with WATER FLOWING NORMALLY (no leak). Press Enter to start WET-NO-LEAK calibration...")
    wet_samples = read_samples_for(ser, CALIBRATION_DURATION_SEC, label="WET-NO-LEAK baseline")

    # Build feature windows from the WET-NO-LEAK calibration (this is our
    # reference "normal operating" state)
    wet_features = []
    n_windows = len(wet_samples) // window_samples
    for i in range(n_windows):
        chunk = wet_samples[i * window_samples:(i + 1) * window_samples]
        wet_features.append(extract_features(chunk, sample_rate))
    wet_features = np.array(wet_features)

    baseline_mean = wet_features.mean(axis=0)
    baseline_cov = np.cov(wet_features.T)
    # regularize covariance slightly to avoid singular matrix issues
    baseline_cov += np.eye(baseline_cov.shape[0]) * 1e-6
    inv_cov = np.linalg.inv(baseline_cov)

    # Establish a distance threshold from the calibration data itself:
    # compute each calibration window's distance to the mean, then set
    # the leak threshold at LEAK_THRESHOLD_STD std-devs above that.
    calib_distances = np.array([
        mahalanobis_distance(f, baseline_mean, inv_cov) for f in wet_features
    ])
    distance_threshold = calib_distances.mean() + LEAK_THRESHOLD_STD * calib_distances.std()

    print("\n=== Calibration complete ===")
    print(f"WET-NO-LEAK baseline: {n_windows} windows, "
          f"mean distance={calib_distances.mean():.2f}, "
          f"std={calib_distances.std():.2f}")
    print(f"Leak alert threshold: distance > {distance_threshold:.2f}")

    print("\n=== Starting live monitoring (Ctrl+C to stop) ===\n")

    buffer = []
    try:
        while True:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 3:
                continue
            try:
                ax, ay, az = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue

            buffer.append((ax, ay, az))

            if len(buffer) >= window_samples:
                chunk = np.array(buffer[:window_samples])
                buffer = buffer[window_samples:]  # slide forward (no overlap here for simplicity)

                live_features = extract_features(chunk, sample_rate)
                distance = mahalanobis_distance(live_features, baseline_mean, inv_cov)

                status = "LEAK DETECTED" if distance > distance_threshold else "normal"
                flag = "  <<<< ALERT" if status == "LEAK DETECTED" else ""
                print(f"[{time.strftime('%H:%M:%S')}] distance={distance:6.2f} "
                      f"(threshold={distance_threshold:.2f}) -> {status}{flag}")

    except KeyboardInterrupt:
        print("\nStopped monitoring.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()