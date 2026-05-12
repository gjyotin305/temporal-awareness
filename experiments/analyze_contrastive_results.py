#!/usr/bin/env python3
"""Analyze contrastive math dataset results.

Loads all per-base-problem .pt files from results/contrastive/ and computes:
  - Final-answer correctness rate per variant
  - Mean top-1 confidence (max softmax prob) per variant, per hop position
  - Mean token-distribution entropy per variant, per hop position
  - Mean generation length (tokens) per variant, per hop position
  - Confidence drop: clean vs error_at_1 (and error_at_2 where available)

Outputs:
  results/contrastive_analysis/summary.json   — aggregate tables
  results/contrastive_analysis/per_hop.json   — per-hop breakdown
  (printed to stdout as well)
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results" / "contrastive"
DATASET_PATH = ROOT / "data" / "raw" / "contrastive_math_dataset.json"
OUT_DIR = ROOT / "results" / "contrastive_analysis"


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────


def mean_top1_prob(logits_dict: dict) -> float:
    """Mean over decoding steps of max-softmax probability (greedy confidence)."""
    total = 0.0
    for step_logits in logits_dict.values():
        total += F.softmax(step_logits.float(), dim=-1).max().item()
    return total / len(logits_dict) if logits_dict else 0.0


def mean_entropy(logits_dict: dict) -> float:
    """Mean Shannon entropy (nats) of the softmax distribution per decoding step."""
    total = 0.0
    for step_logits in logits_dict.values():
        p = F.softmax(step_logits.float(), dim=-1)
        total += -(p * (p + 1e-12).log()).sum().item()
    return total / len(logits_dict) if logits_dict else 0.0


def extract_numbers(text: str) -> list[float]:
    return [float(n) for n in re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))]


def answer_correct(generated_text: str, correct_final_answer: str) -> bool:
    """True if the correct numeric value appears anywhere in the generated text."""
    ref_nums = extract_numbers(correct_final_answer)
    if not ref_nums:
        return False
    gen_nums = extract_numbers(generated_text)
    return any(abs(g - ref_nums[0]) < 0.01 for g in gen_nums)


# ─────────────────────────────────────────────────────────────────────────────
# Load dataset ground truth
# ─────────────────────────────────────────────────────────────────────────────


def load_trace_lookup(dataset_path: Path) -> dict[str, dict]:
    with open(dataset_path) as f:
        data = json.load(f)
    return {t["id"]: t for t in data["traces"]}


# ─────────────────────────────────────────────────────────────────────────────
# Accumulator helpers
# ─────────────────────────────────────────────────────────────────────────────


def _acc() -> dict:
    return {"n": 0, "conf_sum": 0.0, "entropy_sum": 0.0, "len_sum": 0, "correct_n": 0, "correct_total": 0}


def _add(acc: dict, conf: float, entropy: float, n_tokens: int, correct: bool | None) -> None:
    acc["n"] += 1
    acc["conf_sum"] += conf
    acc["entropy_sum"] += entropy
    acc["len_sum"] += n_tokens
    if correct is not None:
        acc["correct_total"] += 1
        if correct:
            acc["correct_n"] += 1


def _summarize(acc: dict) -> dict:
    n = acc["n"] or 1
    ct = acc["correct_total"] or 1
    return {
        "n_hops": acc["n"],
        "mean_confidence": round(acc["conf_sum"] / n, 4),
        "mean_entropy": round(acc["entropy_sum"] / n, 4),
        "mean_gen_length": round(acc["len_sum"] / n, 1),
        "final_answer_accuracy": round(acc["correct_n"] / ct, 4) if acc["correct_total"] else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────


def run_analysis() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset ground truth from {DATASET_PATH}")
    trace_lookup = load_trace_lookup(DATASET_PATH)

    pt_files = sorted(RESULTS_DIR.glob("*.pt"))
    print(f"Found {len(pt_files)} result files")

    # variant -> aggregate accumulator (all hops)
    variant_acc: dict[str, dict] = defaultdict(_acc)
    # variant -> hop_position -> accumulator
    variant_hop_acc: dict[str, dict[int, dict]] = defaultdict(lambda: defaultdict(_acc))
    # hop_depth -> variant -> accumulator
    depth_variant_acc: dict[int, dict[str, dict]] = defaultdict(lambda: defaultdict(_acc))
    # subcategory -> variant -> accumulator
    subcat_variant_acc: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(_acc))

    for pt_path in tqdm(pt_files, desc="Analyzing"):
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        base_id = payload["base_problem_id"]

        for variant, trace_result in payload["traces"].items():
            trace_gt = trace_lookup.get(trace_result["trace_id"], {})
            correct_final = trace_gt.get("correct_final_answer", "")
            subcategory = trace_gt.get("subcategory", "unknown")
            hop_depth = trace_result["hop_depth"]

            for hop in trace_result["hop_results"]:
                if hop["is_injected"]:
                    continue  # no model output to measure

                hi = hop["hop_index"]
                is_last = hi == hop_depth - 1
                logits = hop["logits"]
                gen_text = hop["generated_text"] or ""
                n_tokens = len(logits)

                conf = mean_top1_prob(logits)
                ent = mean_entropy(logits)
                correct = answer_correct(gen_text, correct_final) if is_last else None

                _add(variant_acc[variant], conf, ent, n_tokens, correct)
                _add(variant_hop_acc[variant][hi], conf, ent, n_tokens, correct)
                _add(depth_variant_acc[hop_depth][variant], conf, ent, n_tokens, correct)
                _add(subcat_variant_acc[subcategory][variant], conf, ent, n_tokens, correct)

    # ── Build output dicts ──────────────────────────────────────────────────

    overall = {v: _summarize(a) for v, a in sorted(variant_acc.items())}

    per_hop: dict[str, dict] = {}
    for variant, hop_dict in sorted(variant_hop_acc.items()):
        per_hop[variant] = {f"hop_{hi}": _summarize(a) for hi, a in sorted(hop_dict.items())}

    by_depth: dict[str, dict] = {}
    for depth, vdict in sorted(depth_variant_acc.items()):
        by_depth[str(depth) + "-hop"] = {v: _summarize(a) for v, a in sorted(vdict.items())}

    by_subcat: dict[str, dict] = {}
    for subcat, vdict in sorted(subcat_variant_acc.items()):
        by_subcat[subcat] = {v: _summarize(a) for v, a in sorted(vdict.items())}

    # Confidence deltas (error vs clean)
    deltas: dict[str, dict] = {}
    clean_overall = overall.get("clean", {})
    for variant in ["error_at_1", "error_at_2"]:
        if variant not in overall:
            continue
        err = overall[variant]
        deltas[variant] = {
            "confidence_delta": round(err["mean_confidence"] - clean_overall.get("mean_confidence", 0), 4),
            "entropy_delta": round(err["mean_entropy"] - clean_overall.get("mean_entropy", 0), 4),
            "accuracy_delta": round(
                (err["final_answer_accuracy"] or 0) - (clean_overall.get("final_answer_accuracy") or 0), 4
            ),
        }

    summary = {
        "overall": overall,
        "confidence_deltas_vs_clean": deltas,
        "by_hop_depth": by_depth,
        "by_subcategory": by_subcat,
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "per_hop.json").write_text(json.dumps(per_hop, indent=2))

    # ── Print results ───────────────────────────────────────────────────────
    _print_results(overall, per_hop, by_depth, by_subcat, deltas)
    print(f"\nSaved to {OUT_DIR}/")


def _print_results(overall, per_hop, by_depth, by_subcat, deltas) -> None:
    SEP = "-" * 72

    print("\n" + SEP)
    print("OVERALL RESULTS BY VARIANT")
    print(SEP)
    hdr = f"{'Variant':<14} {'Accuracy':>9} {'Confidence':>11} {'Entropy':>9} {'Gen Len':>8}"
    print(hdr)
    print(SEP)
    for v, s in overall.items():
        acc = f"{s['final_answer_accuracy']:.3f}" if s["final_answer_accuracy"] is not None else "  N/A "
        print(f"{v:<14} {acc:>9} {s['mean_confidence']:>11.4f} {s['mean_entropy']:>9.4f} {s['mean_gen_length']:>8.1f}")

    print("\n" + SEP)
    print("CONFIDENCE DELTAS VS CLEAN")
    print(SEP)
    for variant, d in deltas.items():
        print(f"  {variant}:")
        print(f"    confidence : {d['confidence_delta']:+.4f}")
        print(f"    entropy    : {d['entropy_delta']:+.4f}")
        print(f"    accuracy   : {d['accuracy_delta']:+.4f}")

    print("\n" + SEP)
    print("PER-HOP BREAKDOWN (all variants)")
    print(SEP)
    for variant, hop_dict in per_hop.items():
        print(f"\n  [{variant}]")
        print(f"  {'Hop':<6} {'Accuracy':>9} {'Confidence':>11} {'Entropy':>9} {'Gen Len':>8}")
        for hop_label, s in hop_dict.items():
            acc = f"{s['final_answer_accuracy']:.3f}" if s["final_answer_accuracy"] is not None else "  N/A "
            print(f"  {hop_label:<6} {acc:>9} {s['mean_confidence']:>11.4f} {s['mean_entropy']:>9.4f} {s['mean_gen_length']:>8.1f}")

    print("\n" + SEP)
    print("BY HOP DEPTH")
    print(SEP)
    for depth_label, vdict in by_depth.items():
        print(f"\n  [{depth_label}]")
        print(f"  {'Variant':<14} {'Accuracy':>9} {'Confidence':>11} {'Entropy':>9}")
        for v, s in vdict.items():
            acc = f"{s['final_answer_accuracy']:.3f}" if s["final_answer_accuracy"] is not None else "  N/A "
            print(f"  {v:<14} {acc:>9} {s['mean_confidence']:>11.4f} {s['mean_entropy']:>9.4f}")

    print("\n" + SEP)
    print("BY SUBCATEGORY (final-answer accuracy)")
    print(SEP)
    variants_seen = sorted({v for vd in by_subcat.values() for v in vd})
    col_w = 10
    header = f"  {'Subcategory':<28}" + "".join(f"{v[:col_w]:>{col_w}}" for v in variants_seen)
    print(header)
    print("  " + "-" * (28 + col_w * len(variants_seen)))
    for subcat, vdict in sorted(by_subcat.items()):
        row = f"  {subcat:<28}"
        for v in variants_seen:
            s = vdict.get(v, {})
            acc = s.get("final_answer_accuracy")
            row += f"{'N/A' if acc is None else f'{acc:.3f}':>{col_w}}"
        print(row)
    print()


if __name__ == "__main__":
    run_analysis()
