# scripts/compute_reason_scores.py

import json
import numpy as np
import torch
from train_sae import SparseAutoencoder

REASONING_KEYWORDS = {
    "think", "reason", "because", "step", "so", "therefore", "let", "assume", "define", "if", "then", "we", "proof", "consider"
}

def is_reasoning_token(token):
    return token.lower().strip(".,!?") in REASONING_KEYWORDS

def compute_reason_score(json_path, sae_model_path, feature_count, output_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Prepare full matrix of activations and R/¬R masks
    r_acts, nr_acts = [], []

    input_dim = len(data[0]["activations"][0])
    model = SparseAutoencoder(input_dim)
    model.load_state_dict(torch.load(sae_model_path, map_location="cpu"))
    model.eval()

    for item in data:
        tokens = item["text"].split()
        acts = np.array(item["activations"])

        if len(tokens) != len(acts):
            continue  # skip misaligned

        with torch.no_grad():
            x = torch.tensor(acts, dtype=torch.float32)
            _, h = model(x)

        for token, vec in zip(tokens, h):
            (r_acts if is_reasoning_token(token) else nr_acts).append(vec.numpy())

    r_acts = np.array(r_acts)
    nr_acts = np.array(nr_acts)

    print(f"R tokens: {len(r_acts)}, ¬R tokens: {len(nr_acts)}")

    # Mean activation per feature
    mu_r = r_acts.mean(axis=0)
    mu_nr = nr_acts.mean(axis=0)

    # Normalize scores
    norm_r = mu_r / (mu_r.sum() + 1e-8)
    norm_nr = mu_nr / (mu_nr.sum() + 1e-8)

    reason_score = norm_r - norm_nr

    # Top features
    topk = np.argsort(-reason_score)[:20]
    print("\nTop 20 Reasoning Features:")
    for idx in topk:
        print(f"Feature {idx}: Score = {reason_score[idx]:.6f}")

    # Save scores
    np.save(output_path, reason_score)
    print(f"Saved ReasonScore to {output_path}")

if __name__ == "__main__":
    compute_reason_score(
        json_path="../data/activations_with_text.json",
        sae_model_path="../models/sae_tinyllama.pt",
        feature_count=2048 * 16,
        output_path="../metrics/reason_score.npy"
    )
