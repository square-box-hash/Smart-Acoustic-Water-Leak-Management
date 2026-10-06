"""
Loader for the Aghashahi, Sela & Banks (2023) leak-simulation dataset.

Handles all three sensor types:
  - Accelerometer            (CSV, Sample/Value columns)
  - Dynamic Pressure Sensor  (CSV, Sample/Value columns)
  - Hydrophone                (raw PCM, 8000 Hz, 32-bit, via soundfile)

Labels are parsed from the folder path + filename, since no separate
metadata file is provided. Returns one tidy DataFrame with one row
per trial (not per sample) -- the raw signal itself is stored as a
list/array in a 'signal' column, plus a 'sample_rate' column.

Usage:
    from load_dataset import load_all_trials
    df = load_all_trials(root_path)
    df.to_pickle("trials.pkl")   # cache so you don't re-parse every run
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import soundfile as sf
except ImportError:
    sf = None  # only needed for hydrophone .raw files


# ---------------------------------------------------------------------------
# Label parsing
# ---------------------------------------------------------------------------

NETWORK_CODES = {"BR": "Branched", "LO": "Looped"}
LEAK_CODES = {
    "CC": "Circumferential Crack",
    "GL": "Gasket Leak",
    "LC": "Longitudinal Crack",
    "NL": "No Leak",
    "OL": "Orifice Leak",
}

# Matches things like: BR_OL_0.47 LPS_A1   or   BR_CC_ND_NN_H1  or  BR_CC_Transient_N_H2
FILENAME_RE = re.compile(
    r"^(?P<network>BR|LO)_"
    r"(?P<leak>CC|GL|LC|NL|OL)_"
    r"(?P<state>[\d.]+ ?LPS|ND|Transient)"
    r"(?:_(?P<extra>N|NN))?"
    r"_(?P<channel>[APH]\d+)$"
)


def parse_label(stem: str) -> dict:
    """Parse a filename stem (no extension) into label fields.
    Falls back to raw stem in 'raw_name' if the pattern doesn't match
    (e.g. 'Background Noise_H1')."""
    m = FILENAME_RE.match(stem)
    if not m:
        return {
            "network": None,
            "leak_type": None,
            "state": None,
            "extra_flag": None,
            "channel": stem.rsplit("_", 1)[-1] if "_" in stem else None,
            "raw_name": stem,
        }
    d = m.groupdict()
    return {
        "network": NETWORK_CODES.get(d["network"], d["network"]),
        "leak_type": LEAK_CODES.get(d["leak"], d["leak"]),
        "state": d["state"],
        "extra_flag": d["extra"],
        "channel": d["channel"],
        "raw_name": stem,
    }


# ---------------------------------------------------------------------------
# Signal loading
# ---------------------------------------------------------------------------

def load_csv_signal(path: Path):
    """Accelerometer / Dynamic Pressure Sensor CSVs: columns Sample,Value.

    Sample rate is NOT inferred from the 'Sample' (time) column -- the
    text-precision of that column (e.g. '3.91E-05') is too coarse and
    produces a noisy, wrong estimate (~25601.6 Hz). The dataset's paper
    (Aghashahi, Sela & Banks, 2023, Data in Brief) states the true
    acquisition rate explicitly: 51.2 kHz for both the accelerometer and
    dynamic pressure sensor. We use that documented value instead.
    """
    df = pd.read_csv(path)
    v = df["Value"].to_numpy(dtype=float)
    fs = 51200.0
    return v, fs


def load_raw_signal(path: Path, samplerate=8000, channels=1, subtype="PCM_32", endian="LITTLE"):
    """Hydrophone .raw files: headerless PCM audio."""
    if sf is None:
        raise ImportError("pip install soundfile --break-system-packages")
    signal, _ = sf.read(
        str(path), channels=channels, samplerate=samplerate,
        subtype=subtype, endian=endian,
    )
    return signal, samplerate


# ---------------------------------------------------------------------------
# Folder walkers
# ---------------------------------------------------------------------------

def load_csv_sensor_folder(folder: Path, sensor_name: str) -> pd.DataFrame:
    """Walks an Accelerometer or Dynamic Pressure Sensor folder tree."""
    rows = []
    for path in folder.rglob("*.csv"):
        stem = path.stem
        label = parse_label(stem)
        signal, fs = load_csv_signal(path)
        rows.append({
            "sensor": sensor_name,
            "file_path": str(path),
            "n_samples": len(signal),
            "sample_rate_hz": fs,
            "signal": signal,
            **label,
        })
    return pd.DataFrame(rows)


def load_hydrophone_folder(folder: Path, samplerate=8000) -> pd.DataFrame:
    """Walks the Hydrophone folder tree (raw PCM files)."""
    rows = []
    for path in folder.rglob("*.raw"):
        stem = path.stem
        label = parse_label(stem)
        signal, fs = load_raw_signal(path, samplerate=samplerate)
        rows.append({
            "sensor": "Hydrophone",
            "file_path": str(path),
            "n_samples": len(signal),
            "sample_rate_hz": fs,
            "signal": signal,
            **label,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def load_all_trials(root: str) -> pd.DataFrame:
    root = Path(root)
    frames = []

    accel_dir = root / "Accelerometer" / "Accelerometer"
    if accel_dir.exists():
        frames.append(load_csv_sensor_folder(accel_dir, "Accelerometer"))

    pressure_dir = root / "DynamicPressureSensor" / "Dynamic Pressure Sensor"
    if pressure_dir.exists():
        frames.append(load_csv_sensor_folder(pressure_dir, "DynamicPressureSensor"))

    hydro_dir = root / "Hydrophone"
    if hydro_dir.exists():
        frames.append(load_hydrophone_folder(hydro_dir))

    if not frames:
        raise FileNotFoundError(
            f"No sensor folders found under {root} -- check folder names/paths."
        )

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    ROOT = (
        r"C:\Users\user\Downloads\Dataset of Leak Simulations in Experimental "
        r"Testbed Water Distribution System\Dataset of Leak Simulations in "
        r"Experimental Testbed Water Distribution System"
    )
    df = load_all_trials(ROOT)
    print(df.shape)
    print(df[["sensor", "network", "leak_type", "state", "channel"]].head(20))
    print("\nUnmatched filenames (check these manually):")
    print(df[df["network"].isna()][["sensor", "raw_name"]])

    df.to_pickle("trials.pkl")
    print("\nSaved to trials.pkl")