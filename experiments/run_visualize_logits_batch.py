import argparse
import glob
from pathlib import Path

from visualize_logits_across_tokens import MODEL_NAME, TOP_K, visualize_pt_file


DEFAULT_RESULTS_GLOB = "/data/b22ai063/.mech_interp/temporal-awareness/results/*_math_reasoning.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run logits diagnostics for multiple saved .pt files."
    )
    parser.add_argument(
        "pt_paths",
        nargs="*",
        type=Path,
        help="Explicit .pt files to process. If omitted, --glob is used.",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default=DEFAULT_RESULTS_GLOB,
        help="Glob pattern used when no explicit pt_paths are passed.",
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
        help="How many salient tokens to plot per response.",
    )
    return parser.parse_args()


def resolve_pt_paths(pt_paths: list[Path], glob_pattern: str) -> list[Path]:
    if pt_paths:
        return pt_paths
    return sorted(Path(path) for path in glob.glob(glob_pattern))


def main() -> None:
    args = parse_args()
    pt_paths = resolve_pt_paths(args.pt_paths, args.glob_pattern)

    if not pt_paths:
        raise FileNotFoundError("No .pt files found to visualize.")

    total_plots = 0
    for pt_path in pt_paths:
        print(f"Processing {pt_path}")
        saved_paths = visualize_pt_file(
            pt_path=pt_path,
            model_name=args.model_name,
            top_k=args.top_k,
        )
        total_plots += len(saved_paths)

    print(f"Finished. Generated {total_plots} plots across {len(pt_paths)} .pt files.")


if __name__ == "__main__":
    main()
