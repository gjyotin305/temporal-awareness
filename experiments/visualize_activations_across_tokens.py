from pathlib import Path

import matplotlib.pyplot as plt
import torch

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError:
    PCA = None
    TSNE = None


PT_PATH = Path(
    "/data/b22ai063/.mech_interp/temporal-awareness/results/1_math_reasoning.pt"
)
RESPONSE_KEY = "response_2"
OUTPUT_2D_PATH = Path(
    f"/data/b22ai063/.mech_interp/temporal-awareness/results/1_math_reasoning_{RESPONSE_KEY}_steps_2d.png"
)
OUTPUT_1D_PATH = Path(
    f"/data/b22ai063/.mech_interp/temporal-awareness/results/1_math_reasoning_{RESPONSE_KEY}_steps_component_1.png"
)
OUTPUT_3D_PATH = Path(
    f"/data/b22ai063/.mech_interp/temporal-awareness/results/1_math_reasoning_{RESPONSE_KEY}_steps_3d.png"
)
OUTPUT_3D_PATH_PLANE = Path(
    f"/data/b22ai063/.mech_interp/temporal-awareness/results/1_math_reasoning_{RESPONSE_KEY}_steps_3d_plane.png"
)
OUTPUT_SPECTROGRAM_PATH = Path(
    f"/data/b22ai063/.mech_interp/temporal-awareness/results/1_math_reasoning_{RESPONSE_KEY}_spectrogram.png"
)


def collect_layer_norms(activations: dict) -> tuple[list[int], list[int], torch.Tensor]:
    """Build a [n_steps, n_layers] matrix of last-token hidden-state L2 norms.

    For each generation step and each transformer layer, takes the hidden state
    at the last (most recent) token position and computes its L2 norm.
    This is the closest proxy to "what the model is computing" at that step.

    Returns:
        step_ids: sorted list of generation step indices
        layer_ids: sorted list of layer indices found in the data
        norms: float tensor of shape [n_steps, n_layers]
    """
    step_ids = sorted(activations.keys())
    layer_keys = sorted(
        [k for k in activations[step_ids[0]].keys() if k.startswith("layer_")],
        key=lambda k: int(k.split("_")[1]),
    )
    layer_ids = [int(k.split("_")[1]) for k in layer_keys]

    norms = []
    for step_id in step_ids:
        step_norms = []
        for key in layer_keys:
            hidden = activations[step_id][key].float()  # [1, seq_len, d_model]
            last_token = hidden[0, -1, :]               # [d_model]
            step_norms.append(last_token.norm().item())
        norms.append(step_norms)

    return step_ids, layer_ids, torch.tensor(norms)  # [n_steps, n_layers]


def plot_spectrogram(
    step_ids: list[int],
    layer_ids: list[int],
    norms: torch.Tensor,
    output_path: Path,
) -> None:
    """Spectrogram-style heatmap: x=step_id, y=layer, color=activation L2 norm."""
    matrix = norms.numpy().T  # [n_layers, n_steps] for imshow (y=layer, x=step)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap="inferno",
        interpolation="nearest",
        extent=(step_ids[0] - 0.5, step_ids[-1] + 0.5, layer_ids[0] - 0.5, layer_ids[-1] + 0.5),
    )
    fig.colorbar(im, ax=ax, label="Last-token hidden state L2 norm")

    tick_positions = _xticks(step_ids)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(t) for t in tick_positions])

    n_layers = len(layer_ids)
    ytick_step = max(1, n_layers // 10)
    ax.set_yticks(layer_ids[::ytick_step])

    ax.set_xlabel("Step ID")
    ax.set_ylabel("Layer")
    ax.set_title("Activation spectrogram — last-token L2 norm per layer per step")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def collect_step_vectors(activations: dict) -> tuple[list[int], torch.Tensor]:
    step_ids = sorted(activations.keys())
    vectors = []
    for step_id in step_ids:
        step_tensor = activations[step_id]["mean_pooled_across_layers_and_sequence"]
        vectors.append(step_tensor.squeeze(0).float())
    return step_ids, torch.stack(vectors, dim=0)


def reduce_vectors(
    vectors: torch.Tensor,
    method: str = "pca",
    n_components: int = 3,
) -> torch.Tensor:
    if vectors.ndim != 2:
        raise ValueError(f"Expected [n_steps, hidden], got {tuple(vectors.shape)}")

    n_steps = vectors.shape[0]
    if n_steps < 2:
        raise ValueError("Need at least 2 token steps to reduce and plot activations")

    n_components = min(n_components, n_steps, vectors.shape[1])
    matrix = vectors.cpu().numpy()

    if method == "tsne":
        if TSNE is None:
            raise ImportError("scikit-learn is required for t-SNE visualization")
        perplexity = min(30, max(2, n_steps - 1))
        reduced = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=0,
        ).fit_transform(matrix)
    else:
        if PCA is None:
            # Fall back to torch SVD-style PCA if sklearn is unavailable.
            centered = vectors - vectors.mean(dim=0, keepdim=True)
            _, _, v = torch.pca_lowrank(centered, q=n_components)
            return centered @ v[:, :n_components]
        reduced = PCA(n_components=n_components, random_state=0).fit_transform(matrix)

    return torch.from_numpy(reduced).float()


def _xticks(step_ids: list[int]) -> list[int]:
    """Return a readable subset of step_ids for x-axis tick marks."""
    n = len(step_ids)
    tick_step = max(1, n // 15)
    ticks = step_ids[::tick_step]
    if step_ids[-1] not in ticks:
        ticks = list(ticks) + [step_ids[-1]]
    return ticks


def plot_2d(step_ids: list[int], reduced: torch.Tensor, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    x = step_ids
    plt.plot(x, reduced[:, 0].numpy(), label="Component 1", linewidth=2)
    plt.plot(x, reduced[:, 1].numpy(), label="Component 2", linewidth=2)
    plt.xticks(_xticks(step_ids))
    plt.xlabel("Step ID")
    plt.ylabel("Reduced activation value")
    plt.title("Reduced activations across token steps")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_3d_plane(step_ids: list[int], reduced: torch.Tensor, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    x = step_ids
    plt.plot(x, reduced[:, 0].numpy(), label="Component 1", linewidth=2)
    plt.plot(x, reduced[:, 1].numpy(), label="Component 2", linewidth=2)
    plt.plot(x, reduced[:, 2].numpy(), label="Component 3", linewidth=2)
    plt.xticks(_xticks(step_ids))
    plt.xlabel("Step ID")
    plt.ylabel("Reduced activation value")
    plt.title("Reduced activations across token steps")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_1d_component(step_ids: list[int], reduced: torch.Tensor, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    x = step_ids
    plt.plot(x, reduced[:, 0].numpy(), linewidth=2)
    plt.xticks(_xticks(step_ids))
    plt.xlabel("Step ID")
    plt.ylabel("Component 1")
    plt.title("Component 1 across token steps")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_3d(step_ids: list[int], reduced: torch.Tensor, output_path: Path) -> None:
    """3D trajectory in PC1-PC2-PC3 space, colored by step_id."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        reduced[:, 0].numpy(),
        reduced[:, 1].numpy(),
        reduced[:, 2].numpy(),
        c=step_ids,
        cmap="viridis",
        s=40,
    )
    ax.plot(
        reduced[:, 0].numpy(),
        reduced[:, 1].numpy(),
        reduced[:, 2].numpy(),
        alpha=0.4,
        linewidth=1,
    )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.set_title("PCA trajectory (color = step ID)")
    fig.colorbar(scatter, label="Step ID")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main(method: str = "pca") -> None:
    obj = torch.load(PT_PATH, map_location="cpu")
    response = obj[RESPONSE_KEY]
    activations = response["activations"]

    step_ids, step_vectors = collect_step_vectors(activations)
    reduced_1d = reduce_vectors(step_vectors, method=method, n_components=1)
    plot_1d_component(step_ids, reduced_1d, OUTPUT_1D_PATH)

    reduced_2d = reduce_vectors(step_vectors, method=method, n_components=2)
    plot_2d(step_ids, reduced_2d, OUTPUT_2D_PATH)

    reduced_3d = reduce_vectors(step_vectors, method=method, n_components=3)
    plot_3d(step_ids, reduced_3d, OUTPUT_3D_PATH)

    reduced_3d_plane = reduce_vectors(step_vectors, method=method, n_components=3)
    plot_3d_plane(step_ids, reduced_3d_plane, OUTPUT_3D_PATH_PLANE)

    step_ids_spec, layer_ids, norms = collect_layer_norms(activations)
    plot_spectrogram(step_ids_spec, layer_ids, norms, OUTPUT_SPECTROGRAM_PATH)

    print(f"Saved 1D plot to {OUTPUT_1D_PATH}")
    print(f"Saved 2D plot to {OUTPUT_2D_PATH}")
    print(f"Saved 3D plot to {OUTPUT_3D_PATH}")
    print(f"Saved 3D_PLANE plot to {OUTPUT_3D_PATH_PLANE}")
    print(f"Saved spectrogram to {OUTPUT_SPECTROGRAM_PATH}")


if __name__ == "__main__":
    main(method="pca")
