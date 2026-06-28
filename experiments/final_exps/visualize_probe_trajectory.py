"""
visualize_probe_trajectory.py
=============================
Token-position probe trajectory for 3 clean + 3 error examples.

Two output files, each using its own probe trained on that aggregation mode:
  act_last — probe_last applied to raw hidden state at each token position
  act_mean — probe_mean applied to cumulative mean of hidden states up to each position

X-axis = token index (all tokens shown).
Binary prediction strip at the bottom of each panel.

Examples are drawn from stage1_records.csv (probe training set).

Reads:
    <output_dir>/probe_weights_last.npz
    <output_dir>/probe_weights_mean.npz
    <output_dir>/probe_threshold.json

Usage:
    python visualize_probe_trajectory.py \\
        --model  Qwen/Qwen2.5-3B-Instruct \\
        --output_dir results_qwen25_3b_instruct/ \\
        [--dataset data/raw/contrastive_math_dataset.json] \\
        [--n_examples 3] \\
        [--think]

Outputs:
    <output_dir>/figures/revamp_token_act_last.pdf
    <output_dir>/figures/revamp_token_act_mean.pdf
"""

import argparse
import json
import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASET_DEFAULT = "/home/palashdas/jyotin/.temp_exp/temporal-awareness/data/raw/contrastive_math_dataset.json"
MAX_INPUT_LEN   = 768

VERBALIZED_SYSTEM_PROMPT = (
    "You are solving a multi-step math problem. After each answer, "
    "write your confidence on its own line in the exact format:\n"
    "Confidence: N/10\n"
    "where N is 0 (completely unsure) to 10 (certain)."
)

INK      = "#1a1a1a"
PAPER    = "#fefdf9"
EMERALD  = "#2a7a4a"
CRIMSON  = "#a02a2a"
ASH      = "#8a8580"
AZURE    = "#3a5f8a"
ACCENT   = "#c14a1d"


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset",    default=DATASET_DEFAULT)
    p.add_argument("--n_examples", type=int, default=3)
    p.add_argument("--think",      action="store_true")
    return p.parse_args()


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {name} ...")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    mdl.eval()
    print(f"Loaded: {mdl.config.num_hidden_layers} layers, d={mdl.config.hidden_size}")
    return tok, mdl


# ── Probe ─────────────────────────────────────────────────────────────────────

def load_probes(output_dir):
    out = Path(output_dir)
    with open(out / "probe_threshold.json") as f:
        thresh = json.load(f)
    probes = {}
    for mode in ("last", "mean"):
        pw = np.load(out / f"probe_weights_{mode}.npz", allow_pickle=True)
        t  = thresh[mode]
        probes[mode] = {
            "W":        pw["W"],
            "b":        float(pw["b"]),
            "mu":       pw["scaler_mean"],
            "sc":       pw["scaler_scale"],
            "layer":    int(pw["layer"]),
            "tau":      float(t["threshold"]),
            "tau_cons": float(t["conservative_threshold"]),
        }
    return probes

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def score_last(hs, probe):
    """Per-token score using raw hidden state at each position.
    hs: (L+1, S, d) → (S,)
    """
    h = hs[probe["layer"]]                              # (S, d)
    x = (h - probe["mu"]) / (probe["sc"] + 1e-10)
    return _sigmoid(x @ probe["W"] + probe["b"])        # (S,)

def score_mean(hs, probe):
    """Per-token score using cumulative mean of hidden states up to each position.
    hs: (L+1, S, d) → (S,)
    """
    h    = hs[probe["layer"]]                           # (S, d)
    cumm = np.cumsum(h, axis=0) / (np.arange(1, len(h) + 1)[:, None])  # (S, d)
    x    = (cumm - probe["mu"]) / (probe["sc"] + 1e-10)
    return _sigmoid(x @ probe["W"] + probe["b"])        # (S,)


# ── Prompt building ───────────────────────────────────────────────────────────

def _has_chat_template(tok):
    return getattr(tok, "chat_template", None) is not None

def format_messages(messages, tok, enable_thinking=False):
    if _has_chat_template(tok):
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if enable_thinking:
            kwargs["enable_thinking"] = True
        return tok.apply_chat_template(messages, **kwargs)
    parts = []
    for m in messages:
        parts.append(m["content"] + ("\n\n" if m["role"] == "system" else "\n"))
    return "".join(parts) + "Answer:"

def _resolve(hop):
    if hop["is_injected"] and hop.get("injected_response"):
        return hop["injected_response"]
    return hop["correct_response"]

def build_hop_messages(trace, hop_idx):
    msgs = []
    for i, hop in enumerate(trace["hops"]):
        if i < hop_idx:
            msgs.append({"role": "user",      "content": f"Step {i+1}: {hop['prompt']}"})
            msgs.append({"role": "assistant", "content": f"Answer: {_resolve(hop)}"})
        else:
            msgs.append({"role": "user", "content": f"Step {i+1}: {hop['prompt']}"})
            break
    return msgs


# ── Activation extraction ─────────────────────────────────────────────────────

def get_hidden_states(model, tok, prompt):
    """Single forward pass → hs (L+1, S, d) and decoded token strings."""
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt", max_length=MAX_INPUT_LEN, truncation=True,
              add_special_tokens=not _has_chat_template(tok))
    ids = enc["input_ids"][0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        hs  = torch.stack(out.hidden_states, dim=0)[:, 0, :, :].float().cpu().numpy()
        del out

    torch.cuda.empty_cache()
    gc.collect()

    tokens = tok.convert_ids_to_tokens(ids)
    tokens = [t.replace("▁", " ").replace("Ġ", " ").replace("Ċ", "↵") for t in tokens]
    return hs, tokens


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(panels, probe, out_dir, act_mode):
    """
    panels: list of (title, scores: (S,), tokens: [str], color)
    """
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "pdf.fonttype": 42,
    })

    n = len(panels)
    max_tokens = max(len(tokens) for _, _, tokens, _ in panels)
    fig_w = max(22, max_tokens * 0.17)

    fig, axes = plt.subplots(n, 1, figsize=(fig_w, 4 * n), facecolor=PAPER, squeeze=False)
    axes = axes[:, 0]

    mode_label = ("raw hidden state" if act_mode == "last"
                  else "cumulative mean hidden state")
    fig.suptitle(f"Token-position probe trajectory  ({act_mode}: {mode_label})",
                 fontsize=10, color=INK, y=1.002)

    for ax, (title, scores, tokens, title_color) in zip(axes, panels):
        ax.set_facecolor(PAPER)
        x    = np.arange(len(scores))
        pred = scores >= probe["tau"]

        # Score line — neutral color for all panels
        ax.plot(x, scores, color=AZURE, lw=1.1, zorder=3)

        # Fill: red where above threshold, green where below
        ax.fill_between(x, probe["tau"], scores, where= pred,
                        color=CRIMSON, alpha=0.25, zorder=2)
        ax.fill_between(x, probe["tau"], scores, where=~pred,
                        color=EMERALD, alpha=0.20, zorder=2)

        # Threshold lines
        ax.axhline(probe["tau"],      color=ACCENT, ls="--", lw=0.9,
                   label=f"τ_youden={probe['tau']:.3f}")
        ax.axhline(probe["tau_cons"], color=AZURE,  ls=":",  lw=0.9,
                   label=f"τ_cons={probe['tau_cons']:.3f}")
        ax.axhline(0.5, color=ASH, ls=":", lw=0.4)

        # Binary prediction strip below y=0
        ax.fill_between(x, -0.10, -0.02, where= pred,  color=CRIMSON, alpha=0.75, zorder=4)
        ax.fill_between(x, -0.10, -0.02, where=~pred,  color=EMERALD, alpha=0.45, zorder=4)
        ax.text(-0.5, -0.06, "pred", fontsize=6, color=ASH, va="center", ha="right")

        ax.set_ylim(-0.14, 1.05)
        ax.set_xlim(-1, len(scores))
        ax.set_ylabel("Probe score")
        ax.set_title(title, color=title_color)
        ax.legend(loc="upper right", frameon=False, fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=90, ha="center", fontsize=5)
        ax.tick_params(axis="x", length=3, pad=2)

    plt.tight_layout()
    fig_dir = Path(out_dir) / "figures"
    fig_dir.mkdir(exist_ok=True)
    p = fig_dir / f"revamp_token_act_{act_mode}.pdf"
    fig.savefig(str(p), bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"  -> {p}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _probe_score(hs, probe):
    """Score the last token position using a probe."""
    h = hs[probe["layer"], -1, :]
    x = (h - probe["mu"]) / (probe["sc"] + 1e-10)
    return float(1.0 / (1.0 + np.exp(-(x @ probe["W"] + probe["b"]))))


def main():
    args = get_args()
    probes = load_probes(args.output_dir)
    for mode, p in probes.items():
        print(f"Probe ({mode}): L{p['layer']}  τ={p['tau']:.4f}  τ_cons={p['tau_cons']:.4f}")

    with open(args.dataset) as f:
        dataset = json.load(f)
    by_base = defaultdict(list)
    for t in dataset["traces"]:
        by_base[t["base_problem_id"]].append(t)

    records     = pd.read_csv(Path(args.output_dir) / "stage1_records.csv")
    trained_ids = list(records["base_id"].unique())

    eligible = [
        bid for bid in trained_ids
        if any(v["variant"] == "clean"      for v in by_base[bid])
        and any(v["variant"] == "error_at_1" for v in by_base[bid])
    ]
    clean_ids = eligible[: args.n_examples]
    error_ids = eligible[args.n_examples: args.n_examples * 2]

    print(f"Clean examples: {clean_ids}")
    print(f"Error examples: {error_ids}")

    tok, mdl = load_model(args.model)

    examples = []
    for bid in clean_ids:
        trace  = next(v for v in by_base[bid] if v["variant"] == "clean")
        prompt = format_messages(build_hop_messages(trace, 1), tok, args.think)
        hs, tokens = get_hidden_states(mdl, tok, prompt)
        for mode, p in probes.items():
            score = _probe_score(hs, p)
            print(f"CLEAN  {bid}  [{mode}]  score={score:.3f}  pred={'ERROR' if score >= p['tau'] else 'clean'}")
        examples.append((hs, tokens, "clean", bid, trace["hop_depth"], EMERALD))

    for bid in error_ids:
        trace  = next(v for v in by_base[bid] if v["variant"] == "error_at_1")
        prompt = format_messages(build_hop_messages(trace, 1), tok, args.think)
        hs, tokens = get_hidden_states(mdl, tok, prompt)
        for mode, p in probes.items():
            score = _probe_score(hs, p)
            print(f"ERROR  {bid}  [{mode}]  score={score:.3f}  pred={'ERROR' if score >= p['tau'] else 'clean'}")
        examples.append((hs, tokens, "error_standard", bid, trace["hop_depth"], CRIMSON))

    # Each act_mode uses its own probe trained on that aggregation
    score_fns = {"last": score_last, "mean": score_mean}

    for act_mode, score_fn in score_fns.items():
        probe = probes[act_mode]
        print(f"\nPlotting act_{act_mode} (probe L{probe['layer']}) ...")
        panels = []
        for hs, tokens, cond, bid, hop_depth, color in examples:
            scores = score_fn(hs, probe)
            label  = "CLEAN" if cond == "clean" else "ERROR"
            title  = f"{label}  |  {bid}  |  hop_depth={hop_depth}  |  act_{act_mode}"
            panels.append((title, scores, tokens, color))
        plot(panels, probe, args.output_dir, act_mode)

    print(f"\nDone. Figures in {args.output_dir}/figures/")


if __name__ == "__main__":
    main()
