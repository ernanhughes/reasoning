import torch
import numpy as np
import os
import yaml

class SparseAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation="relu", sparsity_target=0.03, sparsity_penalty=0.0001):
        super().__init__()
        self.encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.decoder = torch.nn.Linear(hidden_dim, input_dim)
        self.activation_fn = getattr(torch.nn.functional, activation)
        self.sparsity_target = sparsity_target
        self.sparsity_penalty = sparsity_penalty
        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "activation": activation,
            "sparsity_target": sparsity_target,
            "sparsity_penalty": sparsity_penalty
        }

    def forward(self, x):
        z = self.activation_fn(self.encoder(x))
        recon = self.decoder(z)
        return recon, z

    def sparsity_loss(self, z):
        # Approx L1 or KL-divergence regularizer
        return self.sparsity_penalty * torch.mean(torch.abs(z))

    @classmethod
    def from_config(cls, config):
        return cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            activation=config.get("activation", "relu"),
            sparsity_target=config.get("sparsity_target", 0.03),
            sparsity_penalty=config.get("sparsity_penalty", 0.0001)
        )

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        with open(os.path.join(path, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)

    @classmethod
    def load(cls, path, device=None):
        with open(os.path.join(path, "config.yaml"), "r") as f:
            config = yaml.safe_load(f)
        model = cls.from_config(config)
        state_dict = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=device or "cpu")
        model.load_state_dict(state_dict)
        return model

    # def load_sae_model(sae_path):
    #     config_path = os.path.join(sae_path, "config.json")
    #     weights_path = os.path.join(sae_path, "pytorch_model.bin")

    #     # Load config
    #     import json
    #     with open(config_path) as f:
    #         cfg = json.load(f)
    #     input_dim = cfg["input_dim"]
    #     hidden_dim = cfg["hidden_dim"]

    #     # Initialize and load model
    #     model = SparseAutoencoder(input_dim, hidden_dim)
    #     model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    #     model.eval()
    #     return model

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

    def summary(self, print_output=True):
        info = {
            "input_dim": self.config["input_dim"],
            "hidden_dim": self.config["hidden_dim"],
            "activation": self.config["activation"],
            "sparsity_target": self.config["sparsity_target"],
            "sparsity_penalty": self.config["sparsity_penalty"],
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

        if print_output:
            print("🔍 Sparse Autoencoder Summary:")
            for k, v in info.items():
                print(f"  {k}: {v}")

        return info
