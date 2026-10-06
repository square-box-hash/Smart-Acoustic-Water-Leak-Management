"""
Leak size / flow vs detection rate, by leak type.

Reads grouped_predictions.csv (written by grouped_check.py).
Fill LEAK_INFO by hand from the dataset paper: only what the paper actually reports.
Leave None where it does not say. Use the SAME unit for every leak type.

Run:  python flow_detection_table.py
"""
import numpy as np
import pandas as pd

# ---------------- CONFIG: EDIT FROM THE DATASET PAPER ----------------
PRED_CSV = "grouped_predictions.csv"
LEAK_INFO = {
    "Orifice Leak":          {"size": None, "flow": None},
    "Gasket Leak":           {"size": None, "flow": None},
    "Longitudinal Crack":    {"size": None, "flow": None},
    "Circumferential Crack": {"size": None, "flow": None},
}
SIZE_UNIT = "?"   # e.g. mm or mm^2, as the paper states
FLOW_UNIT = "?"   # e.g. L/s, as the paper states
OUT_CSV = "leak_flow_detection.csv"
OUT_PNG = "flow_vs_detection.png"
# ---------------------------------------------------------------------


def main():
    df = pd.read_csv(PRED_CSV)
    leaks = df[df["y"] == 1].copy()

    tab = leaks.groupby("leak_type").agg(
        n_recordings=("y", "size"),
        detection_rate=("y_pred", "mean"),
        mean_vote_leak=("vote_leak_fraction", "mean"),
    ).round(3)
    tab["size"] = [LEAK_INFO.get(t, {}).get("size") for t in tab.index]
    tab["flow"] = [LEAK_INFO.get(t, {}).get("flow") for t in tab.index]
    tab = tab.sort_values("detection_rate")

    print("=== DETECTION BY LEAK TYPE (leave-one-experiment-out predictions) ===")
    print(f"size unit: {SIZE_UNIT}; flow unit: {FLOW_UNIT}")
    print(tab.to_string())
    tab.to_csv(OUT_CSV)
    print(f"\nsaved {OUT_CSV}")

    if "state" in leaks.columns:
        print("\n=== DETECTION RATE BY LEAK TYPE x BACKGROUND FLOW STATE ===")
        print("(each cell is only 4 recordings: 2 networks x 2 channels. Read big gaps only.)")
        piv = leaks.pivot_table(index="leak_type", columns="state",
                                values="y_pred", aggfunc="mean").round(2)
        print(piv.to_string())

    have_flow = tab["flow"].dropna()
    if len(have_flow) >= 3:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nmatplotlib missing: python -m pip install matplotlib")
            return
        d = tab.dropna(subset=["flow"])
        plt.figure(figsize=(7, 5))
        plt.scatter(d["flow"].astype(float), d["detection_rate"], s=120)
        for name, row in d.iterrows():
            plt.annotate(name, (float(row["flow"]), row["detection_rate"]),
                         textcoords="offset points", xytext=(8, 6), fontsize=12)
        plt.xlabel(f"Leak flow ({FLOW_UNIT}), from the dataset paper", fontsize=13)
        plt.ylabel("Detection rate (fraction predicted Leak)", fontsize=13)
        plt.ylim(0, 1.05)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=200)
        print(f"saved {OUT_PNG}")
    else:
        print("\nFill in at least 3 flow values in LEAK_INFO to get the plot.")


if __name__ == "__main__":
    main()