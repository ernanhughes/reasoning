import torch
import numpy as np
import os

class SparseAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.decoder = torch.nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def load_sae_model(sae_path):
    config_path = os.path.join(sae_path, "config.json")
    weights_path = os.path.join(sae_path, "pytorch_model.bin")

    # Load config
    import json
    with open(config_path) as f:
        cfg = json.load(f)
    input_dim = cfg["input_dim"]
    hidden_dim = cfg["hidden_dim"]

    # Initialize and load model
    model = SparseAutoencoder(input_dim, hidden_dim)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

def compute_reason_score_for_layer(hidden_states, sae_model, top_k=20):
    """
    hidden_states: (1, seq_len, hidden_dim) from intermediate layer
    sae_model: trained SparseAutoencoder
    """
    _, z = sae_model(hidden_states.squeeze(0))  # shape: (seq_len, hidden_dim)
    # Compute L1 norm per feature across tokens
    token_scores = torch.abs(z).sum(dim=0)
    # Get top_k activations as measure of reasoning potential
    topk_vals, _ = torch.topk(token_scores, k=top_k)
    return topk_vals.mean().item()
