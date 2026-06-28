"""
collect_activations.py
======================
Minimal script to collect residual-stream activations for probe training.
Runs a single forward pass per prompt (no generation) to extract hidden states.

Reads:
    <dataset>  — JSON with "traces" list (extended_contrastive_math_dataset.json)

Writes:
    <output_dir>/act_last.npy     (N, n_layers+1, d_model)  last-token activations
    <output_dir>/act_mean.npy     (N, n_layers+1, d_model)  mean-over-prompt activations
    <output_dir>/act_labels.npy   (N,)  binary: 0=clean, 1=error
    <output_dir>/stage1_records.csv

For each base_problem_id, three conditions are collected:
    clean            (label=0) — correct prior answer in context
    error_standard   (label=1) — injected wrong answer in context
    error_verbalized (label=1) — same error but with verbalized-confidence system prompt

Usage:
    python collect_activations.py \\
        --dataset data/raw/extended_contrastive_math_dataset.json \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --output_dir results_qwen25_3b_instruct/ \\
        --n_samples 100

    # For thinking models (Qwen3-think, etc.)
    python collect_activations.py ... --think

    # Without 4-bit quantization
    python collect_activations.py ... --no_4bit
"""

import argparse
import gc
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CHECKPOINT_EVERY = 20
MAX_INPUT_LEN = 768

VERBALIZED_SYSTEM_PROMPT = (
    "You are solving a multi-step math problem. After each answer, "
    "write your confidence on its own line in the exact format:\n"
    "Confidence: N/10\n"
    "where N is 0 (completely unsure) to 10 (certain)."
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Collect residual-stream activations for probe training")
    p.add_argument("--dataset",    default="/home/palashdas/jyotin/.temp_exp/temporal-awareness/data/raw/contrastive_math_dataset.json")
    p.add_argument("--model",      required=True, help="HuggingFace model name or local path")
    p.add_argument("--output_dir", required=True, help="Directory to save outputs")
    p.add_argument("--n_samples",  type=int, default=100, help="Number of base problems to process")
    p.add_argument("--think",      action="store_true",   help="Pass enable_thinking=True in chat template (Qwen3-think etc.)")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {name} ...")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    mdl.eval()
    n_layers = mdl.config.num_hidden_layers
    d_model  = mdl.config.hidden_size
    print(f"Loaded: {n_layers} layers, d_model={d_model}")
    return tok, mdl, n_layers, d_model


# ── Prompt building ───────────────────────────────────────────────────────────

def _has_chat_template(tokenizer):
    return getattr(tokenizer, "chat_template", None) is not None

def format_messages(messages, tokenizer, enable_thinking=False):
    if _has_chat_template(tokenizer):
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if enable_thinking:
            kwargs["enable_thinking"] = True
        return tokenizer.apply_chat_template(messages, **kwargs)
    parts = []
    for m in messages:
        if m["role"] == "system":
            parts.append(m["content"] + "\n\n")
        else:
            parts.append(m["content"] + "\n")
    return "".join(parts) + "Answer:"

def tokenize_prompt(tokenizer, prompt):
    return tokenizer(
        prompt, return_tensors="pt",
        max_length=MAX_INPUT_LEN, truncation=True,
        add_special_tokens=not _has_chat_template(tokenizer),
    )

def _resolve_response(hop, overrides, i):
    if overrides and i in overrides:
        return overrides[i]
    if hop["is_injected"] and hop.get("injected_response"):
        return hop["injected_response"]
    return hop["correct_response"]

def build_hop_messages(trace, hop_idx):
    messages = []
    for i, hop in enumerate(trace["hops"]):
        if i < hop_idx:
            resp = _resolve_response(hop, None, i)
            messages.append({"role": "user",      "content": f"Step {i+1}: {hop['prompt']}"})
            messages.append({"role": "assistant", "content": f"Answer: {resp}"})
        else:
            messages.append({"role": "user", "content": f"Step {i+1}: {hop['prompt']}"})
            break
    return messages

def build_verbalized_messages(trace, hop_idx):
    messages = [{"role": "system", "content": VERBALIZED_SYSTEM_PROMPT}]
    for i, hop in enumerate(trace["hops"]):
        if i < hop_idx:
            resp = _resolve_response(hop, None, i)
            messages.append({"role": "user",      "content": f"Step {i+1}: {hop['prompt']}"})
            messages.append({"role": "assistant", "content": f"Answer: {resp}\nConfidence: 10/10"})
        else:
            messages.append({"role": "user", "content": f"Step {i+1}: {hop['prompt']}"})
            break
    return messages


# ── Activation extraction (single forward pass, no generation) ────────────────

def extract_acts(model, tokenizer, prompt):
    """Forward pass only — returns (acts_last, acts_mean, prompt_len).
    acts_last: (n_layers+1, d_model)  — last-token hidden state per layer
    acts_mean: (n_layers+1, d_model)  — mean over all prompt tokens per layer
    """
    device = next(model.parameters()).device
    enc = tokenize_prompt(tokenizer, prompt)
    enc = {k: v.to(device) for k, v in enc.items()}
    S = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        hs  = torch.stack(out.hidden_states, dim=0)  # (n_layers+1, 1, S, d)
        acts_last = hs[:, 0, S - 1, :].float().cpu().numpy()  # (L+1, d)
        acts_mean = hs[:, 0,  :,   :].float().mean(dim=1).cpu().numpy()  # (L+1, d)
        del out, hs

    del enc
    torch.cuda.empty_cache()
    gc.collect()
    return acts_last, acts_mean, S


# ── Save helpers ──────────────────────────────────────────────────────────────

def checkpoint(out, records, act_last_list, act_mean_list, labels):
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / "stage1_records.csv", index=False)
    np.save(out / "act_last.npy",   np.stack(act_last_list, axis=0))
    np.save(out / "act_mean.npy",   np.stack(act_mean_list, axis=0))
    np.save(out / "act_labels.npy", np.array(labels))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)

    with open(args.dataset) as f:
        dataset = json.load(f)

    by_base = defaultdict(list)
    for t in dataset["traces"]:
        by_base[t["base_problem_id"]].append(t)

    all_ids = list(by_base.keys())
    random.shuffle(all_ids)
    sampled = all_ids[: args.n_samples]
    print(f"Dataset: {len(dataset['traces'])} traces, {len(all_ids)} base problems")
    print(f"Collecting {args.n_samples} base problems → {args.n_samples * 3} activation sets\n")

    tok, mdl, n_layers, d_model = load_model(args.model)

    act_last_list, act_mean_list, act_labels = [], [], []
    records = []
    t0 = time.time()

    for idx, base_id in enumerate(sampled):
        variants = by_base[base_id]
        clean = next((t for t in variants if t["variant"] == "clean"),       None)
        error = next((t for t in variants if t["variant"] == "error_at_1"), None)
        if not clean or not error:
            continue

        elapsed = time.time() - t0
        eta     = (elapsed / max(idx, 1)) * (len(sampled) - idx)
        print(f"[{idx+1:3d}/{len(sampled)}] {base_id} ({clean['hop_depth']}h)  "
              f"{elapsed/60:.0f}m  ETA {eta/60:.0f}m", flush=True)

        conditions = [
            ("clean",            clean, format_messages(build_hop_messages(clean, 1), tok, args.think), 0),
            ("error_standard",   error, format_messages(build_hop_messages(error, 1), tok, args.think), 1),
            ("error_verbalized", error, format_messages(build_verbalized_messages(error, 1), tok, args.think), 1),
        ]

        for cond_name, trace, prompt, label in conditions:
            acts_last, acts_mean, prompt_len = extract_acts(mdl, tok, prompt)
            act_last_list.append(acts_last)
            act_mean_list.append(acts_mean)
            act_labels.append(label)
            records.append({
                "base_id":    base_id,
                "condition":  cond_name,
                "label":      label,
                "hop_depth":  trace["hop_depth"],
                "error_type": trace.get("injected_error_type"),
                "prompt_len": prompt_len,
            })

        if (idx + 1) % CHECKPOINT_EVERY == 0:
            checkpoint(out, records, act_last_list, act_mean_list, act_labels)
            print(f"  [checkpoint at {idx+1}]")

    checkpoint(out, records, act_last_list, act_mean_list, act_labels)

    total = time.time() - t0
    print(f"\nDone in {total/60:.1f} min")
    print(f"  act_last.npy   {np.stack(act_last_list).shape}")
    print(f"  act_mean.npy   {np.stack(act_mean_list).shape}")
    print(f"  act_labels.npy {np.array(act_labels).shape}")
    print(f"  stage1_records.csv  ({len(records)} rows)")
    print(f"Outputs in: {out}/")


if __name__ == "__main__":
    main()
