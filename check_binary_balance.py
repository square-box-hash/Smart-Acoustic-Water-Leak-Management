"""
Binary class balance check (No-Leak vs pooled Leak), matching the real
trials.pkl schema:

  columns: sensor, file_path, n_samples, sample_rate_hz, signal, network,
           leak_type, state, extra_flag, channel, raw_name

Each row = one full trial (one raw CSV recording). There is no separate
trial_id column -- raw_name (or the DataFrame index) uniquely identifies
a trial.
"""

import pandas as pd

TRIALS_PKL = "trials.pkl"

df = pd.read_pickle(TRIALS_PKL)
print(f"Loaded {len(df)} rows (= trials) from {TRIALS_PKL}\n")

# --- check the NaN leak_type rows first -- don't silently pool these ---
nan_rows = df[df["leak_type"].isna()]
print(f"Rows with NaN leak_type: {len(nan_rows)}")
if len(nan_rows):
    print(nan_rows[["sensor", "network", "state", "raw_name"]].to_string())
    print()

# --- per-sensor binary balance (excluding NaN rows) ---
clean = df.dropna(subset=["leak_type"])
clean = clean.copy()
clean["binary_label"] = clean["leak_type"].apply(
    lambda x: "No-Leak" if x == "No Leak" else "Leak"
)

print("=== Per-sensor binary trial counts ===")
for sensor, group in clean.groupby("sensor"):
    counts = group["binary_label"].value_counts()
    print(f"\n{sensor}: {len(group)} trials")
    print(counts.to_string())
    total = counts.sum()
    print(f"  Leak ratio: {counts.get('Leak', 0) / total:.1%}")

print("\n=== Raw leak-type breakdown (pre-pooling, all sensors) ===")
print(clean["leak_type"].value_counts().to_string())

print("\n=== Accelerometer-only leak-type breakdown ===")
accel = clean[clean["sensor"] == "Accelerometer"]
print(accel["leak_type"].value_counts().to_string())