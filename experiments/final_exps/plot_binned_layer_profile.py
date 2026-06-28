"""
plot_binned_layer_profile.py
============================
Compresses the per-layer AUROC profile into three stage bins (Start / Middle / End)
and draws a two-line chart (act_last vs act_mean) with ±1σ error bands.

Usage:
    python plot_binned_layer_profile.py --output_dir results_qwen25_3b_instruct/

Output:
    <output_dir>/figures/fig_binned_layer_profile.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK     = "#1a1a1a"
PAPER   = "#fefdf9"
EMERALD = "#2a7a4a"
AZURE   = "#3a5f8a"
ASH     = "#8a8580"

BIN_LABELS = ["Start", "Middle", "End"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def compute_bins(df: pd.DataFrame):
    n_layers = df["layer"].max() + 1
    edges = [
        (0,           n_layers // 3),
        (n_layers // 3, 2 * n_layers // 3),
        (2 * n_layers // 3, n_layers),
    ]
    rows = []
    for mode in ("last", "mean"):
        sub = df[df["mode"] == mode]
        for label, (lo, hi) in zip(BIN_LABELS, edges):
            bucket = sub[(sub["layer"] >= lo) & (sub["layer"] < hi)]
            rows.append({
                "mode":  mode,
                "bin":   label,
                "auroc": bucket["auroc_mean"].mean(),
                "std":   bucket["auroc_mean"].std(),
            })
    return pd.DataFrame(rows)


def plot(binned: pd.DataFrame, out_dir: Path):
    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "legend.fontsize":    9,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.8,
        "pdf.fonttype":       42,
    })

    fig, ax = plt.subplots(figsize=(5, 3.8), facecolor=PAPER)
    ax.set_facecolor(PAPER)

    x = np.arange(len(BIN_LABELS))

    style = {
        "last": dict(color=AZURE,    marker="o", ms=6, lw=1.6, label="act_last"),
        "mean": dict(color=EMERALD,  marker="s", ms=6, lw=1.6, label="act_mean"),
    }

    for mode, kw in style.items():
        d = binned[binned["mode"] == mode].set_index("bin").loc[BIN_LABELS]
        auroc = d["auroc"].values
        std   = d["std"].values

        ax.plot(x, auroc, **kw, zorder=3)
        ax.fill_between(x, auroc - std, auroc + std,
                        color=kw["color"], alpha=0.15, zorder=2)

    ax.axhline(0.5, color=ASH, ls=":", lw=0.8, label="chance")

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_ylabel("AUROC (mean ± 1σ within bin)")
    ax.set_xlabel("Layer stage")
    ax.set_title("Error-encoding probe AUROC by layer stage")
    ax.set_ylim(0.45, 1.02)
    ax.legend(frameon=False, loc="lower right")

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    out_path = fig_dir / "fig_binned_layer_profile.pdf"
    fig.savefig(str(out_path), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)

    df = pd.read_csv(out_dir / "layer_probe_results.csv")
    binned = compute_bins(df)
    print(binned.to_string(index=False))
    plot(binned, out_dir)


if __name__ == "__main__":
    main()
