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
RESPONSE_KEY = "response_0"
OUTPUT_2D_PATH = Path(
    "/data/b22ai063/.mech_interp/temporal-awareness/results/1_math_reasoning_response_0_steps_2d.png"
)
OUTPUT_1D_PATH = Path(
    "/data/b22ai063/.mech_interp/temporal-awareness/results/1_math_reasoning_response_0_steps_component_1.png"
)
OUTPUT_3D_PATH = Path(
    "/data/b22ai063/.mech_interp/temporal-awareness/results/1_math_reasoning_response_0_steps_3d.png"
)
OUTPUT_3D_PATH_PLANE = Path(
    "/data/b22ai063/.mech_interp/temporal-awareness/results/1_math_reasoning_response_0_steps_3d_plane.png"
)


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


def plot_2d(step_ids: list[int], reduced: torch.Tensor, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    x = torch.tensor(step_ids)
    plt.plot(x.numpy(), reduced[:, 0].numpy(), label="Component 1", linewidth=2)
    plt.plot(x.numpy(), reduced[:, 1].numpy(), label="Component 2", linewidth=2)
    plt.xlabel("Token step")
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
    x = torch.tensor(step_ids)
    
    plt.plot(x.numpy(), reduced[:, 0].numpy(), label="Component 1", linewidth=2)
    plt.plot(x.numpy(), reduced[:, 1].numpy(), label="Component 2", linewidth=2)
    plt.plot(x.numpy(), reduced[:, 2].numpy(), label="Component 3", linewidth=2)

    plt.xlabel("Token step")
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
    x = torch.tensor(step_ids)
    plt.plot(x.numpy(), reduced[:, 0].numpy(), linewidth=2)
    plt.xlabel("Token step")
    plt.ylabel("Component 1")
    plt.title("Component 1 across token steps")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_3d(step_ids: list[int], reduced: torch.Tensor, output_path: Path) -> None:
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    x = torch.tensor(step_ids)
    scatter = ax.scatter(
        x.numpy(),
        reduced[:, 0].numpy(),
        reduced[:, 1].numpy(),
        c=step_ids,
        cmap="viridis",
        s=40,
    )
    ax.plot(
        x.numpy(),
        reduced[:, 0].numpy(),
        reduced[:, 1].numpy(),
        alpha=0.4,
        linewidth=1,
    )
    ax.set_xlabel("Token step")
    ax.set_ylabel("Component 1")
    ax.set_zlabel("Component 2")
    ax.set_title("Reduced activations across token steps (3D)")
    fig.colorbar(scatter, label="Token step")
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

    print(f"Saved 1D plot to {OUTPUT_1D_PATH}")
    print(f"Saved 2D plot to {OUTPUT_2D_PATH}")
    print(f"Saved 3D plot to {OUTPUT_3D_PATH}")
    print(f"Saved 3D_PLANE plot to {OUTPUT_3D_PATH_PLANE}")


if __name__ == "__main__":
    main(method="pca")
