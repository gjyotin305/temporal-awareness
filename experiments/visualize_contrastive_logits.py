#!/usr/bin/env python3
"""Per-question token-wise confidence and entropy plots for contrastive variants.

For each base_problem_id produces one PNG with layout:
  rows   : confidence (top-1 softmax prob) | entropy (Shannon, nats)
  columns: one per hop position

Within each cell every variant (clean / error_at_1 / error_at_2) that
generated tokens at that hop is drawn as a separate line.
Hops that were injected for a variant are shaded and labelled rather than
plotted, so the reader can immediately see where the wrong context was planted.

Output: results/contrastive_plots/<base_problem_id>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results" / "contrastive"
OUT_DIR = ROOT / "results" / "contrastive_plots"

VARIANT_STYLE: dict[str, dict] = {
    "clean":      {"color": "#2196F3", "lw": 2.0, "alpha": 0.9, "zorder": 3},
    "error_at_1": {"color": "#F44336", "lw": 1.8, "alpha": 0.85, "zorder": 2},
    "error_at_2": {"color": "#FF9800", "lw": 1.8, "alpha": 0.85, "zorder": 2},
}
INJECT_COLORS: dict[str, str] = {
    "error_at_1": "#FFCDD2",
    "error_at_2": "#FFE0B2",
}
VARIANT_ORDER = ["clean", "error_at_1", "error_at_2"]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────


def logits_to_probs(logits_dict: dict) -> torch.Tensor:
    """Return float Tensor [n_steps, vocab] of softmax probabilities."""
    steps = sorted(logits_dict.keys())
    return torch.stack(
        [F.softmax(logits_dict[s].float().squeeze(0), dim=-1) for s in steps], dim=0
    )


def top1_confidence(probs: torch.Tensor) -> torch.Tensor:
    """[n_steps] max probability at each decoding step."""
    return probs.max(dim=-1).values


def entropy(probs: torch.Tensor) -> torch.Tensor:
    """[n_steps] Shannon entropy (nats) at each decoding step."""
    return -(probs * (probs + 1e-12).log()).sum(dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Plot one base problem
# ─────────────────────────────────────────────────────────────────────────────


def plot_base_problem(payload: dict, out_path: Path) -> None:
    base_id: str = payload["base_problem_id"]
    traces: dict = payload["traces"]

    # Determine max hop depth across all variants
    hop_depth = max(t["hop_depth"] for t in traces.values())

    fig, axes = plt.subplots(
        2, hop_depth,
        figsize=(5 * hop_depth, 6),
        sharex="col",
        squeeze=False,
    )

    fig.suptitle(base_id, fontsize=11, fontweight="bold", y=1.01)

    # Label rows
    axes[0][0].set_ylabel("Confidence\n(top-1 prob)", fontsize=9)
    axes[1][0].set_ylabel("Entropy (nats)", fontsize=9)

    for hop_idx in range(hop_depth):
        ax_conf = axes[0][hop_idx]
        ax_ent  = axes[1][hop_idx]
        ax_conf.set_title(f"Hop {hop_idx}", fontsize=9)
        ax_ent.set_xlabel("Token step", fontsize=8)

        for variant in VARIANT_ORDER:
            if variant not in traces:
                continue
            trace = traces[variant]
            if hop_idx >= len(trace["hop_results"]):
                continue
            hop = trace["hop_results"][hop_idx]
            style = VARIANT_STYLE[variant]

            if hop["is_injected"]:
                # Shade the cell to indicate injection
                for ax in (ax_conf, ax_ent):
                    ax.set_facecolor(INJECT_COLORS.get(variant, "#EEEEEE"))
                    ax.text(
                        0.5, 0.5,
                        f"{variant}\n(injected)",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=8, color="#555555", style="italic",
                    )
            else:
                logits = hop.get("logits")
                if not logits:
                    continue
                probs = logits_to_probs(logits)        # [T, vocab]
                conf  = top1_confidence(probs).numpy() # [T]
                ent   = entropy(probs).numpy()          # [T]
                steps = list(range(len(conf)))

                ax_conf.plot(steps, conf, label=variant,
                             color=style["color"], lw=style["lw"],
                             alpha=style["alpha"], zorder=style["zorder"])
                ax_ent.plot(steps, ent, label=variant,
                            color=style["color"], lw=style["lw"],
                            alpha=style["alpha"], zorder=style["zorder"])

        for ax in (ax_conf, ax_ent):
            ax.grid(alpha=0.25, linewidth=0.6)
            ax.tick_params(labelsize=7)

        ax_conf.set_ylim(0, 1.05)
        ax_ent.set_ylim(bottom=0)

    # Shared legend
    legend_patches = [
        mpatches.Patch(color=VARIANT_STYLE[v]["color"], label=v)
        for v in VARIANT_ORDER
        if v in traces
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(legend_patches),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Token-wise confidence/entropy plots for contrastive math variants."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR,
        help="Directory containing per-base-problem .pt files (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_DIR,
        help="Output directory for PNG files (default: %(default)s)",
    )
    parser.add_argument(
        "--base-problem-ids", nargs="+", default=None, metavar="ID",
        help="Restrict to specific base_problem_ids (default: all)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip problems whose PNG already exists",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N problems (useful for quick checks)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.base_problem_ids:
        pt_files = [args.results_dir / f"{bid}.pt" for bid in args.base_problem_ids]
    else:
        pt_files = sorted(args.results_dir.glob("*.pt"))

    if args.limit:
        pt_files = pt_files[: args.limit]

    print(f"Plotting {len(pt_files)} base problems → {args.out_dir}")

    for pt_path in tqdm(pt_files, desc="Plotting"):
        out_path = args.out_dir / f"{pt_path.stem}.png"
        if args.skip_existing and out_path.exists():
            continue
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        plot_base_problem(payload, out_path)

    print("Done.")


if __name__ == "__main__":
    main()
