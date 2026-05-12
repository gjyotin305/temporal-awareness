#!/usr/bin/env python3
"""Run inference on contrastive math dataset, collecting per-hop logits.

For each base_problem_id, processes all variants (clean, error_at_1, error_at_2).
  - Injected hops use the pre-set injected_response as the assistant turn.
  - Non-injected hops run model generation and record per-step logits.

Output: one .pt file per base_problem_id under results/contrastive/<base_problem_id>.pt

File schema:
  {
    "base_problem_id": str,
    "traces": {
      "<variant>": {
        "trace_id": str,
        "variant": str,
        "hop_depth": int,
        "injected_error_type": str | None,
        "injected_error_hop": int | None,
        "hop_results": [
          {
            "hop_index": int,
            "prompt": str,
            "is_injected": bool,
            # injected hops — no generation
            "injected_response": str | None,
            # generated hops
            "generated_text": str | None,
            "generated_token_ids": list[int] | None,
            # dict[step_index -> Tensor(1, vocab)] next-token logits at each decoding step
            "logits": dict | None,
          }, ...
        ]
      }
    }
  }
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "data" / "raw" / "contrastive_math_dataset.json"
OUTPUT_DIR = ROOT / "results" / "contrastive"

SYS_PROMPT = (
    "You are a careful math reasoning assistant.\n"
    "Solve the problem step by step, keep the reasoning internally consistent,\n"
    "and give the final answer clearly.\n"
    "If the prompt contains supporting context, use it only when it is relevant."
)

DEFAULT_MODEL = "unsloth/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 256


# ─────────────────────────────────────────────────────────────────────────────
# Generation helpers
# ─────────────────────────────────────────────────────────────────────────────


def generate_with_logits(
    input_ids: torch.Tensor,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_tokens: int = MAX_NEW_TOKENS,
) -> tuple[list[int], dict[int, torch.Tensor]]:
    """Greedy decode, returning generated token ids and per-step next-token logits.

    Returns:
        generated_token_ids: list of generated token ids (excluding EOS)
        step_logits: dict mapping decoding step index -> Tensor(1, vocab_size) on CPU
    """
    generated_token_ids: list[int] = []
    step_logits: dict[int, torch.Tensor] = {}
    past_key_values = None
    current_ids = input_ids

    with torch.inference_mode():
        for step in range(max_tokens):
            outputs = model(current_ids, use_cache=True, past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]  # (1, vocab)
            step_logits[step] = next_token_logits.detach().cpu()

            next_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            token = next_id.item()
            if token == tokenizer.eos_token_id:
                break

            generated_token_ids.append(token)
            current_ids = next_id

    return generated_token_ids, step_logits


# ─────────────────────────────────────────────────────────────────────────────
# Per-trace runner
# ─────────────────────────────────────────────────────────────────────────────


def run_trace(
    trace: dict,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> dict:
    """Process a single trace, returning hop_results list."""
    hops: list[dict] = trace["hops"]
    hop_results: list[dict] = []

    # Conversation history accumulated as chat messages for tokenizer template.
    # Starts empty; we'll push user/assistant turns as we go.
    messages: list[dict] = [{"role": "system", "content": SYS_PROMPT}]

    for hop in hops:
        hop_index: int = hop["hop_index"]
        prompt: str = hop["prompt"]
        is_injected: bool = hop["is_injected"]

        # Add the user turn for this hop.
        messages.append({"role": "user", "content": prompt})

        result: dict = {
            "hop_index": hop_index,
            "prompt": prompt,
            "is_injected": is_injected,
            "injected_response": None,
            "generated_text": None,
            "generated_token_ids": None,
            "logits": None,
        }

        if is_injected:
            # Use the pre-set wrong response — no model generation.
            injected_response: str = hop["injected_response"]
            result["injected_response"] = injected_response
            messages.append({"role": "assistant", "content": injected_response})
        else:
            # Generate with the model and record logits.
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(model.device)

            token_ids, logits = generate_with_logits(input_ids, model, tokenizer)
            generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)

            result["generated_text"] = generated_text
            result["generated_token_ids"] = token_ids
            result["logits"] = logits

            # Add generated response to history so subsequent hops see it.
            messages.append({"role": "assistant", "content": generated_text})

        hop_results.append(result)

    return {
        "trace_id": trace["id"],
        "variant": trace["variant"],
        "hop_depth": trace["hop_depth"],
        "injected_error_type": trace.get("injected_error_type"),
        "injected_error_hop": trace.get("injected_error_hop"),
        "hop_results": hop_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading & grouping
# ─────────────────────────────────────────────────────────────────────────────


def load_and_group(dataset_path: Path) -> dict[str, list[dict]]:
    """Return traces grouped by base_problem_id."""
    with open(dataset_path) as f:
        data = json.load(f)

    groups: dict[str, list[dict]] = defaultdict(list)
    for trace in data["traces"]:
        groups[trace["base_problem_id"]].append(trace)
    return dict(groups)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on contrastive math dataset.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="HuggingFace model id or local path (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write result .pt files (default: %(default)s)",
    )
    parser.add_argument(
        "--base-problem-ids",
        nargs="+",
        default=None,
        metavar="ID",
        help="Restrict to specific base_problem_ids (default: all)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip base_problem_ids whose output file already exists",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Max tokens to generate per hop (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: auto-detect)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {DATASET_PATH}")
    groups = load_and_group(DATASET_PATH)

    target_ids: list[str] = (
        args.base_problem_ids if args.base_problem_ids is not None else sorted(groups.keys())
    )
    print(f"Processing {len(target_ids)} base problem(s).")

    print(f"Loading model {args.model} on {args.device}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for base_problem_id in tqdm(target_ids, desc="Base problems"):
        out_path = args.output_dir / f"{base_problem_id}.pt"
        if args.skip_existing and out_path.exists():
            continue

        traces = groups.get(base_problem_id, [])
        if not traces:
            print(f"Warning: no traces found for {base_problem_id!r}")
            continue

        variant_results: dict[str, dict] = {}
        for trace in tqdm(traces, desc=base_problem_id, leave=False):
            variant = trace["variant"]
            variant_results[variant] = run_trace(trace, model, tokenizer)

        payload = {
            "base_problem_id": base_problem_id,
            "traces": variant_results,
        }
        torch.save(payload, out_path)

    print(f"Done. Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
