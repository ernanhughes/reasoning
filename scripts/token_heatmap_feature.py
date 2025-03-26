import json
import torch
import numpy as np
from termcolor import colored
from train_sae import SparseAutoencoder

def score_to_color(score, max_score):
    # Normalize score [0, 1], then map to color
    norm = score / max_score if max_score else 0
    if norm > 0.66:
        return "red"
    elif norm > 0.33:
        return "yellow"
    else:
        return "white"

def visualize_token_heatmap(json_path, sae_model_path, feature_idx, sample_idx=0):
    # Load activations
    with open(json_path, "r") as f:
        data = json.load(f)

    item = data[sample_idx]
    tokens = item["text"].split()  # crude tokenization
    act_matrix = np.array(item["activations"])  # shape: (seq_len, d_model)

    # Load SAE
    model = SparseAutoencoder(input_dim=act_matrix.shape[1])
    model.load_state_dict(torch.load(sae_model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        X = torch.tensor(act_matrix, dtype=torch.float32)
        _, H = model(X)  # (seq_len, hidden_dim)

    feature_scores = H[:, feature_idx].numpy()
    max_score = feature_scores.max()

    print(f"\n=== Token-level activation for Feature {feature_idx} ===\n")
    for tok, score in zip(tokens, feature_scores):
        col = score_to_color(score, max_score)
        print(colored(f"{tok} [{score:.2f}]", col, attrs=["bold"]), end=' ')
    print()

if __name__ == "__main__":
    visualize_token_heatmap(
        json_path="../data/activations_with_text.json",
        sae_model_path="../models/sae_tinyllama.pt",
        feature_idx=17456 % (2048 * 16),
        sample_idx=0
    )
