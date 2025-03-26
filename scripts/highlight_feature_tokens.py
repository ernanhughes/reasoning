import json
import torch
import numpy as np
from termcolor import colored
from train_sae import SparseAutoencoder

def highlight_tokens_for_feature(
    json_path, sae_model_path, feature_idx, top_k=10
):
    # Load activation data
    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples from {json_path}")

    # Stack activations and keep index map
    all_tokens, all_hidden = [], []
    for idx, item in enumerate(data):
        act = np.array(item["activations"])  # shape (seq_len, dim)
        all_hidden.append(act)
        all_tokens.append(item["text"].split())  # rough token split for now

    all_hidden = np.concatenate(all_hidden, axis=0)  # (N, d)
    input_dim = all_hidden.shape[1]

    # Load SAE
    model = SparseAutoencoder(input_dim=input_dim)
    model.load_state_dict(torch.load(sae_model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        X = torch.tensor(all_hidden, dtype=torch.float32)
        _, H = model(X)  # (N, hidden_dim)

    feature_values = H[:, feature_idx].numpy()

    # Top K positions
    top_indices = np.argsort(-feature_values)[:top_k]

    # Map top indices back to sample/token
    print(f"\nTop {top_k} tokens activating Feature {feature_idx}:\n")
    current_pos = 0
    for i, (tokens, act_matrix) in enumerate(zip(all_tokens, data)):
        seq_len = len(tokens)
        for j in range(seq_len):
            global_idx = current_pos + j
            if global_idx in top_indices:
                # Highlight this token
                highlighted = [
                    colored(tok, "green", attrs=["bold"]) if k == j else tok
                    for k, tok in enumerate(tokens)
                ]
                print(f"Text {i:03d}: {' '.join(highlighted)}")
                break
        current_pos += seq_len

if __name__ == "__main__":
    highlight_tokens_for_feature(
        json_path="../data/activations_with_text.json",
        sae_model_path="../models/sae_tinyllama.pt",
        feature_idx=17456 % (2048 * 16),
        top_k=20
    )
