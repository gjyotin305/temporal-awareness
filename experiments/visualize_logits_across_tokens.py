from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer


MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct"
PT_PATH = Path(
    "/data/b22ai063/.mech_interp/temporal-awareness/results/1_math_reasoning.pt"
)
RESPONSE_KEYS = ("response_0", "response_1", "response_2")
TOP_K = 10


def collect_step_logits(logits_by_step: dict) -> tuple[list[int], torch.Tensor]:
    """Return a [n_steps, vocab] tensor of next-token logits across generation."""
    step_ids = sorted(logits_by_step.keys())
    step_logits = []

    for step_id in step_ids:
        logits = logits_by_step[step_id].float()
        if logits.ndim == 3:
            logits = logits[:, -1, :]
        elif logits.ndim != 2:
            raise ValueError(
                f"Expected logits with shape [B, vocab] or [B, seq, vocab], got {tuple(logits.shape)}"
            )
        step_logits.append(logits.squeeze(0))

    return step_ids, torch.stack(step_logits, dim=0)


def select_top_k_token_ids(step_probs: torch.Tensor, top_k: int) -> torch.Tensor:
    """Pick tokens that are most salient anywhere in the trajectory."""
    if step_probs.ndim != 2:
        raise ValueError(f"Expected [n_steps, vocab], got {tuple(step_probs.shape)}")

    peak_prob_per_token = step_probs.max(dim=0).values
    k = min(top_k, peak_prob_per_token.numel())
    return torch.topk(peak_prob_per_token, k=k).indices


def format_token_label(tokenizer: AutoTokenizer, token_id: int) -> str:
    token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    token_text = token_text.replace("\n", "\\n")
    if not token_text:
        token_text = "<empty>"
    if len(token_text) > 18:
        token_text = token_text[:15] + "..."
    return f"{repr(token_text)} ({token_id})"


def compute_entropy(step_probs: torch.Tensor) -> torch.Tensor:
    return -(step_probs * step_probs.clamp_min(1e-12).log()).sum(dim=-1)


def compute_top2_margin(step_probs: torch.Tensor) -> torch.Tensor:
    top2_probs = torch.topk(step_probs, k=min(2, step_probs.shape[-1]), dim=-1).values
    if top2_probs.shape[-1] < 2:
        return top2_probs[:, 0]
    return top2_probs[:, 0] - top2_probs[:, 1]


def plot_token_prob_diagnostics(
    step_ids: list[int],
    step_probs: torch.Tensor,
    token_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    output_path: Path,
) -> None:
    entropy = compute_entropy(step_probs)
    margin = compute_top2_margin(step_probs)
    tick_positions = step_ids[:: max(1, len(step_ids) // 15)] if step_ids else []

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 11),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1, 1]},
    )

    ax_probs, ax_entropy, ax_margin = axes
    top_token_ids = token_ids.tolist()
    highlighted_token_ids = set(top_token_ids[: min(5, len(top_token_ids))])

    for token_id in top_token_ids:
        is_highlighted = token_id in highlighted_token_ids
        ax_probs.plot(
            step_ids,
            step_probs[:, token_id].numpy(),
            linewidth=2.4 if is_highlighted else 1.2,
            alpha=0.95 if is_highlighted else 0.35,
            label=format_token_label(tokenizer, token_id),
        )

    ax_probs.set_ylabel("Token probability")
    ax_probs.set_title(
        f"Top-{len(token_ids)} next-token probability trajectories across generation"
    )
    ax_probs.grid(alpha=0.3)
    ax_probs.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax_entropy.plot(step_ids, entropy.numpy(), color="tab:orange", linewidth=2.2)
    ax_entropy.set_ylabel("Entropy")
    ax_entropy.set_title("Next-token entropy")
    ax_entropy.grid(alpha=0.3)

    ax_margin.plot(step_ids, margin.numpy(), color="tab:green", linewidth=2.2)
    ax_margin.set_xlabel("Generation step")
    ax_margin.set_ylabel("Top1 - Top2")
    ax_margin.set_title("Confidence margin between the best two tokens")
    ax_margin.grid(alpha=0.3)
    ax_margin.set_xticks(tick_positions)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_output_path(pt_path: Path, response_key: str, top_k: int) -> Path:
    return pt_path.with_name(
        f"{pt_path.stem}_{response_key}_token_prob_diagnostics_top_{top_k}.png"
    )


def plot_response(
    response: dict,
    response_key: str,
    tokenizer: AutoTokenizer,
    output_path: Path,
    top_k: int,
) -> None:
    logits_by_step = response["activations"]

    step_ids, step_logits = collect_step_logits(logits_by_step)
    step_probs = torch.softmax(step_logits, dim=-1)
    token_ids = select_top_k_token_ids(step_probs, top_k=top_k)
    plot_token_prob_diagnostics(step_ids, step_probs, token_ids, tokenizer, output_path)

    print(f"Saved {response_key} token-probability plot to {output_path}")


def visualize_pt_file(
    pt_path: Path,
    model_name: str = MODEL_NAME,
    top_k: int = TOP_K,
    response_keys: tuple[str, ...] = RESPONSE_KEYS,
) -> list[Path]:
    obj = torch.load(pt_path, map_location="cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    saved_paths = []

    for response_key in response_keys:
        if response_key not in obj:
            print(f"Skipping {response_key} for {pt_path.name}: key not found")
            continue
        output_path = build_output_path(pt_path, response_key, top_k)
        plot_response(
            response=obj[response_key],
            response_key=response_key,
            tokenizer=tokenizer,
            output_path=output_path,
            top_k=top_k,
        )
        saved_paths.append(output_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize next-token probability diagnostics for all responses in a .pt result file."
    )
    parser.add_argument(
        "--pt-path",
        type=Path,
        default=PT_PATH,
        help="Path to a saved *_math_reasoning.pt file.",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Tokenizer model name used to decode token labels.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="How many salient tokens to plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_pt_file(
        pt_path=args.pt_path,
        model_name=args.model_name,
        top_k=args.top_k,
    )
