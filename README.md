# Acoustic Pipe-Leak Detection (NCSC 2026–27, Water sub-theme)

A signal-processing and machine-learning investigation into detecting water-pipe
leaks from accelerometer vibration signals, using the public **Aghashahi, Sela &
Banks (2023)** benchmarking dataset, with a focus on honest validation and a
separate analysis of joint-type (gasket) leaks.

This project was developed for the **National Children's Science Congress
(NCSC) 2026–27**, registered title *"Smart Acoustic Water Leakage Management,"*
Water sub-theme, Senior category.

## Motivation

The project started from observing a continuously leaking pipe joint outside a
neighbour's house — a leak that had gone unnoticed for some time because it was
not visually obvious. Field surveys of 7 residents and 3 local plumbers
confirmed that pipe leaks are common, often take weeks to notice, and that no
one locally uses a dedicated acoustic leak detector. India's urban utilities
also lose an average of ~38% of treated water to non-revenue water, well above
the 15–20% global benchmark — this project is a small, local investigation into
whether low-cost acoustic sensing and simple machine learning can help close
that detection gap, not a claim to solve it at scale.

## Dataset

This repo does **not** include the dataset. It uses the publicly available
benchmarking dataset:

> Aghashahi, M., Sela, L., Banks, M.K. (2023). *Benchmarking dataset for leak
> detection and localization in water distribution systems.* Data in Brief, 48,
> 109148. https://doi.org/10.1016/j.dib.2023.109148
> Data: https://data.mendeley.com/datasets/tbrnp6vrnj/1

Download the dataset and run `load_dataset.py` to produce `trials.pkl` (~1.5 GB,
not tracked in this repo) before running anything else.

The dataset: a 47 m, 152.4 mm PVC testbed (looped and branched topologies), with
four induced leak types — Orifice, Longitudinal Crack, Circumferential Crack,
Gasket Leak — plus a No-Leak condition, recorded by accelerometer, hydrophone,
and dynamic pressure sensors across varying background flow and noise
conditions.

## Pipeline (frozen)

The main, reported result uses:
- **Sensor:** Accelerometer (2 channels, A1/A2), 51.2 kHz native, decimated ×10
  to 5120 Hz.
- **Windowing:** 3 s windows, 50% overlap, per 30 s recording.
- **Features:** Continuous wavelet transform (Morlet, 20 log-spaced scales,
  ~2–2000 Hz), mean power per scale per window.
- **Classifier:** SVM (RBF kernel), features standardized per fold.
- **Validation:** Leave-one-experiment-out — each experiment (network × leak
  type × background flow condition) contributes two accelerometer recordings
  (A1, A2); both are held out together to avoid the partner channel leaking
  into training. A recording's label is the majority vote of its window-level
  predictions.
- **Class weights:** `{No-Leak: 4, Leak: 1}` for the binary Leak-vs-No-Leak
  task (chosen after inspecting results — reported as a limitation, not a
  blind choice); `class_weight="balanced"` for the joint-leak tasks.

> **Note on naming:** the feature extractor is a continuous wavelet transform
> (CWT) with Morlet wavelets, not a wavelet *scattering* transform (WST),
> despite some file/variable names referencing WST. `kymatio`-based scattering
> was attempted but is broken against the installed scipy version, so the
> fallback CWT path (`USE_KYMATIO = False`) is what was actually used for all
> reported results.

## Headline results (binary Leak vs No-Leak, accelerometer)

| Validation | Accuracy | Balanced acc. | No-Leak recall | Leak recall | Permutation p |
|---|---|---|---|---|---|
| Leave-one-**recording**-out | 0.812 | 0.742 | 0.625 | 0.859 | 0.0099 |
| Leave-one-**experiment**-out (stricter) | 0.800 | 0.734 | 0.625 | 0.844 | 0.0198 |

Accuracy alone is close to the 80% majority-class baseline (dataset is 80%
Leak / 20% No-Leak); **balanced accuracy and the permutation test are the
evidence the result is above chance**, not accuracy alone.

A feature-ablation and cross-network transfer check (`tiny_leakage_test.py`)
found: the signal is **not** a loudness artifact (total-energy-only feature ≈
chance), is concentrated in the 20–200 Hz band, and does **not** transfer well
across network topologies when trained on only one (near-chance balanced
accuracy), indicating the model partly relies on network-specific structure.

## Joint-leak (gasket) analysis

A separate, motivated side-analysis (see `joint_leaks.py`) asking whether
joint-type leaks (the dataset's Gasket Leak class) are acoustically detectable
and distinguishable from cut-type leaks (Orifice, Circumferential Crack,
Longitudinal Crack), using the same frozen pipeline.

| Task | Balanced accuracy | Permutation p | Notes |
|---|---|---|---|
| Gasket vs No-Leak | 0.812 | 0.0099 | 13/16 each class correct; significant |
| Gasket vs cut-type leaks | 0.594 | 0.21 | Gasket recall only 5/16; **not significant** |

Detection rate by leak type does not track measured leak flow rate (from the
dataset paper's Table 2): Gasket has the *highest* flow (~0.06 L/s) and the
*highest* detection rate (0.94), while Orifice has the second-highest flow
(~0.042 L/s) and the *lowest* detection rate (0.44). See
`mean_spectra_plot.py` for the per-type spectral comparison — notably, crack
leaks show **lower** spectral power than No-Leak across most of the band, an
unexplained and openly reported anomaly.

Sample sizes here are small (8 experiments / 16 recordings per leak type), so
these are exploratory findings, not strong claims.

## Scripts

| Script | Purpose |
|---|---|
| `load_dataset.py` | Loads all sensor folders into one labeled DataFrame, caches to `trials.pkl` |
| `wst_svm_binary_loto.py` | Core pipeline: windowing, CWT feature extraction, binary SVM, naive per-recording LOTO with class-weight sweep |
| `full_results_wst.py` | Full metrics for the frozen pipeline (CI, balanced acc., MCC, ROC-AUC, permutation test, metadata breakdown, class-weight sensitivity table) |
| `grouped_check.py` | Stricter leave-one-**experiment**-out check (both accelerometer channels held out together) |
| `tiny_leakage_test.py` | Feature ablations (loudness vs. shape vs. frequency band) and cross-network transfer test |
| `joint_leaks.py` | Gasket-vs-No-Leak and Gasket-vs-cuts tasks, same frozen pipeline |
| `flow_detection_table.py` | Leak flow rate vs. detection rate table, by leak type and background flow state |
| `mean_spectra_plot.py` | Mean CWT spectrum per leak type, and difference from No-Leak |

Older exploratory scripts (multi-class LOTO-CV, anomaly-score approaches,
hydrophone/pressure-sensor analysis, within-trial time-split diagnostics) are
kept for the project history/logbook record but are not part of the frozen,
reported pipeline above.

## How to run

```powershell
# 1. Place the downloaded dataset folders where load_dataset.py expects them, then:
python load_dataset.py

# 2. Core binary pipeline (first run extracts features; takes a few minutes)
python wst_svm_binary_loto.py

# 3. Full metrics for the frozen pipeline
python full_results_wst.py

# 4. Stricter experiment-level check
python grouped_check.py

# 5. Ablations / cross-network transfer
python tiny_leakage_test.py

# 6. Joint-leak analysis
python joint_leaks.py
python flow_detection_table.py   # fill in LEAK_INFO from the dataset paper's Table 2 first
python mean_spectra_plot.py
```

Requires: `numpy`, `pandas`, `scipy`, `scikit-learn`, `pywt` (PyWavelets),
`matplotlib`.

## Honesty notes (important for anyone reading this as a judge or reviewer)

This project went through several corrections along the way, documented in the
logbook and left visible here deliberately:
- An early 92–100% accuracy was found to be **inflated by data leakage**
  (overlapping windows from the same recording landed on both sides of a
  window-level train/test split). The corrected, trial-level number was
  33–35%.
- Class weights for the binary task were tuned **after** seeing LOTO-CV
  results — reported as a limitation, with a full sensitivity table included
  rather than hidden.
- A proposed low-frequency "drop-check" verifier for the hydrophone sensor,
  which looked promising on a single example, failed to generalize (held in
  only 10.4% of trials when tested systematically) and is reported as a
  negative result.
- The crack-leak spectra being *quieter* than No-Leak is reported as an open,
  unexplained observation rather than something papered over with a
  post-hoc story.

## Limitations

- Lab testbed only (new PVC pipe, open-air, 47 m); not buried, aged, or
  corroded pipe.
- Generalization across network topology is weak when trained on only one
  topology.
- Gasket-leak sample size is small (16 recordings / 8 experiments).
- Hong Kong dataset generalization test and own hardware (INMP441-based)
  validation are planned but not yet complete at time of writing.

## Acknowledgements

Dataset: Aghashahi, Sela & Banks (2023), Data in Brief. 