import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import welch

# Change this only if your original dataset is somewhere else.
DATASET_ROOT = Path(r"C:\Users\user\Downloads\Dataset of Leak Simulations in Experimental Testbed Water Distribution System\Dataset of Leak Simulations in Experimental Testbed Water Distribution System")

ACC_ROOT = DATASET_ROOT / "Accelerometer" / "Accelerometer"

SAMPLE_RATE = 51200

results = []

csv_files = sorted(ACC_ROOT.rglob("*.csv"))

print(f"Found {len(csv_files)} accelerometer CSV files.")

for i, file_path in enumerate(csv_files, start=1):

    try:
        data = pd.read_csv(file_path, usecols=["Value"])
        signal = data["Value"].to_numpy(dtype=np.float64).copy()

        # Remove DC offset
        signal -= np.mean(signal)

        freqs, psd = welch(
            signal,
            fs=SAMPLE_RATE,
            nperseg=min(65536, len(signal)),
            noverlap=min(32768, len(signal) // 2)
        )

        valid = freqs > 1

        freqs_valid = freqs[valid]
        psd_valid = psd[valid]

        # Dominant frequency
        dominant_frequency = freqs_valid[np.argmax(psd_valid)]

        # Cumulative spectral energy
        cumulative = np.cumsum(psd_valid)
        total_energy = cumulative[-1]

        low_idx = np.searchsorted(
            cumulative,
            0.025 * total_energy
        )

        high_idx = np.searchsorted(
            cumulative,
            0.975 * total_energy
        )

        low_freq = freqs_valid[
            min(low_idx, len(freqs_valid) - 1)
        ]

        high_freq = freqs_valid[
            min(high_idx, len(freqs_valid) - 1)
        ]

        results.append({
            "trial_file": str(file_path),
            "dominant_frequency_hz": dominant_frequency,
            "energy_2_5_percent_hz": low_freq,
            "energy_97_5_percent_hz": high_freq
        })

        print(f"[{i}/{len(csv_files)}] {file_path.name}")

    except Exception as e:
        print(f"ERROR: {file_path}")
        print(e)


results_df = pd.DataFrame(results)

print("\n=== DOMINANT FREQUENCY ===")
print(results_df["dominant_frequency_hz"].describe())

print("\n=== 95% SPECTRAL ENERGY RANGE ===")
print(
    results_df[
        ["energy_2_5_percent_hz",
         "energy_97_5_percent_hz"]
    ].describe()
)

print("\n=== HIGHEST 10 DOMINANT FREQUENCIES ===")
print(
    results_df[
        ["trial_file", "dominant_frequency_hz"]
    ]
    .sort_values(
        "dominant_frequency_hz",
        ascending=False
    )
    .head(10)
)

print("\n=== LOWEST 10 DOMINANT FREQUENCIES ===")
print(
    results_df[
        ["trial_file", "dominant_frequency_hz"]
    ]
    .sort_values(
        "dominant_frequency_hz"
    )
    .head(10)
)

results_df.to_csv(
    "frequency_diagnostic.csv",
    index=False
)

print("\nSaved: frequency_diagnostic.csv")