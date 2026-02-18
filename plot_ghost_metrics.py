#!/usr/bin/env python3
"""
Ghost Trajectory Metrics Plotter
==================================
Reads metrics CSV files from all variants and produces a 2×2 subplot figure:
  (0,0) Commanded Linear Velocity  vs Time
  (0,1) Commanded Angular Velocity vs Time
  (1,0) Euclidean Distance to Goal vs Time
  (1,1) Heading Error to Goal      vs Time

Usage:
    cd /path/to/rl_nav
    python3 plot_ghost_metrics.py
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========================== Configuration ==========================
ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "ghost_trajectory_results")
RUN_NAME = "run_1"

VARIANTS = ["SAC_Dense", "SAC_Sparse_HER", "SAC_Penalized", "SAC_Dense_Hist", "SAC_Sparse_HER_Hist"]

# Publication-quality styling
COLORS = {
    "SAC_Dense":          "#1f77b4",
    "SAC_Sparse_HER":     "#ff7f0e",
    "SAC_Penalized":      "#2ca02c",
    "SAC_Dense_Hist":     "#d62728",
    "SAC_Sparse_HER_Hist": "#9467bd",
}

LABELS = {
    "SAC_Dense":          "SAC-Dense",
    "SAC_Sparse_HER":     "SAC-Sparse (HER)",
    "SAC_Penalized":      "SAC-Penalized",
    "SAC_Dense_Hist":     "SAC-Dense + Hist",
    "SAC_Sparse_HER_Hist": "SAC-Sparse (HER) + Hist",
}

DPI = 300
OUTPUT_PATH = os.path.join(RESULTS_DIR, f"{RUN_NAME}_metrics_comparison.png")

# ========================== Load Data ==========================
def load_metrics(variant_name: str) -> dict:
    """Load a metrics CSV and return dict of numpy arrays."""
    csv_path = os.path.join(RESULTS_DIR, variant_name, f"{RUN_NAME}_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"  [SKIP] {variant_name}: CSV not found at {csv_path}")
        return None

    data = {"timestep": [], "v_cmd": [], "w_cmd": [], "d_tg": [], "dtheta_tg": []}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in data:
                data[key].append(float(row[key]))

    return {k: np.array(v) for k, v in data.items()}


# ========================== Plot ==========================
def main():
    plt.rcParams.update({
        # Use LaTeX for all text rendering → produces true Type 1 fonts
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "ps.fonttype": 42,
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    subplot_config = [
        {"ax": axes[0, 0], "key": "v_cmd",     "ylabel": r"$v^{cmd}_t$ [m/s]",  "title": "Commanded Linear Velocity"},
        {"ax": axes[0, 1], "key": "w_cmd",     "ylabel": r"$\omega^{cmd}_t$ [rad/s]", "title": "Commanded Angular Velocity"},
        {"ax": axes[1, 0], "key": "d_tg",      "ylabel": r"$d_{tg}$ [m]",        "title": "Euclidean Distance to Goal"},
        {"ax": axes[1, 1], "key": "dtheta_tg", "ylabel": r"$\Delta\theta_{tg}$ [rad]", "title": "Heading Error to Goal"},
    ]

    for variant in VARIANTS:
        data = load_metrics(variant)
        if data is None:
            continue

        # Drop last observation (often noisy terminal state)
        t = data["timestep"][:-1]
        color = COLORS[variant]
        label = LABELS[variant]

        for sp in subplot_config:
            sp["ax"].plot(t, data[sp["key"]][:-1], color=color, label=label, linewidth=1.2, alpha=0.85)

    for sp in subplot_config:
        ax = sp["ax"]
        ax.set_xlabel("Timestep")
        ax.set_ylabel(sp["ylabel"])
        ax.set_title(sp["title"])
        ax.legend(loc="best")

    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
    pdf_path = OUTPUT_PATH.replace(".png", ".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to: {OUTPUT_PATH}")
    print(f"Saved PDF to: {pdf_path}")


if __name__ == "__main__":
    main()
