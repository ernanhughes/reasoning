import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from train_sae import SparseAutoencoder

def visualize_feature_heatmap(
    activations_path, sae_model_path, feature_idx, top_k=20, save_path=None
):
    print(f"Loading activations from {activations_path}")
    X = torch.tensor(np.load(activations_path), dtype=torch.float32)

    input_dim = X.shape[1]
    model = SparseAutoencoder(input_dim=input_dim)
    model.load_state_dict(torch.load(sae_model_path, map_location="cpu"))
    model.eval()

    print(f"Extracting activations for feature {feature_idx}...")
    with torch.no_grad():
        _, H = model(X)  # shape: (N, hidden_dim)

    # Select top K inputs for this feature
    feature_activations = H[:, feature_idx].numpy()
    top_indices = np.argsort(-feature_activations)[:top_k]
    top_values = feature_activations[top_indices]

    # Plot heatmap
    plt.figure(figsize=(10, 1.5))
    sns.heatmap(top_values.reshape(1, -1), cmap="viridis", annot=True, cbar=True)
    plt.title(f"Top {top_k} Activations for Feature {feature_idx}")
    plt.xlabel("Sample Index")
    plt.yticks([])
    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Example usage
    visualize_feature_heatmap(
        activations_path="../data/activations_tinyllama.npy",
        sae_model_path="../models/sae_tinyllama.pt",
        feature_idx=17456 % (2048 * 16),  # wrap-around just in case
        top_k=20
    )
