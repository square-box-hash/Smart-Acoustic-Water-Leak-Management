"""
Mean spectrum per leak type from the cached CWT scale-power features.

Panel (a): mean log10 power vs approximate centre frequency, one line per leak type.
Panel (b): each leak type minus No Leak (log10 difference), to show where they differ.

Averaging: windows -> recording -> experiment (both channels averaged) -> leak type.
Shaded bands are +/- 1 standard error across the 8 experiments per leak type.

Needs full_results_wst.py and grouped_check.py in the same folder.
Run:  python mean_spectra_plot.py     (writes mean_spectra.png)
"""
import numpy as np
import pandas as pd
import pywt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from full_results_wst import get_features
from grouped_check import build_table
from wst_svm_binary_loto import EFFECTIVE_SAMPLE_RATE, WINDOW_LEN

OUT_PNG = "mean_spectra.png"
BASELINE = "No Leak"


def main():
    windows_df, features = get_features()
    ids = windows_df["trial_id"].values.astype(str)
    tbl = build_table(windows_df)

    scales = np.geomspace(2, WINDOW_LEN / 8, num=features.shape[1])
    freqs = pywt.scale2frequency("morl", scales) * EFFECTIVE_SAMPLE_RATE
    order = np.argsort(freqs)
    f_sorted = freqs[order]

    logp = np.log10(features + 1e-12)
    cols = [f"f{i}" for i in range(logp.shape[1])]
    df_w = pd.DataFrame(logp, columns=cols)
    df_w["trial_id"] = ids

    rec = df_w.groupby("trial_id").mean()
    rec = rec.join(tbl.set_index("trial_id")[["leak_type", "experiment"]])
    exp = rec.groupby(["leak_type", "experiment"]).mean()
    g = exp.groupby(level="leak_type")
    mean = g.mean()
    se = g.std() / np.sqrt(g.size().values[:, None])

    print("experiments per leak type:", g.size().to_dict())

    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    for t in mean.index:
        y = mean.loc[t].values[order]
        e = se.loc[t].values[order]
        style = "--" if t == BASELINE else "-"
        ax.plot(f_sorted, y, style, lw=2.5, marker="o", ms=4, label=t)
        ax.fill_between(f_sorted, y - e, y + e, alpha=0.15)
    ax.set_xscale("log")
    ax.axvspan(20, 200, color="grey", alpha=0.12)
    ax.set_xlabel("Approx. centre frequency (Hz, log scale)")
    ax.set_ylabel("Mean log10 power (arbitrary units)")
    ax.set_title("(a) Mean spectrum by leak type")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    ax = axes[1]
    if BASELINE in mean.index:
        for t in mean.index:
            if t == BASELINE:
                continue
            d = (mean.loc[t] - mean.loc[BASELINE]).values[order]
            e = np.sqrt(se.loc[t] ** 2 + se.loc[BASELINE] ** 2).values[order]
            ax.plot(f_sorted, d, lw=2.5, marker="o", ms=4, label=t)
            ax.fill_between(f_sorted, d - e, d + e, alpha=0.15)
        ax.axhline(0, color="black", lw=1)
        ax.set_xscale("log")
        ax.axvspan(20, 200, color="grey", alpha=0.12)
        ax.set_xlabel("Approx. centre frequency (Hz, log scale)")
        ax.set_ylabel("log10 power minus No Leak")
        ax.set_title("(b) Difference from No Leak")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()