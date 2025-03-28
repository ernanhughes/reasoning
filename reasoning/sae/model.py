import torch
import torch.nn
from pathlib import Path
import yaml
from typing import Dict, Any, Tuple, Union
from torch import Tensor


class SparseAutoencoder(torch.nn.Module):
    """
    Sparse Autoencoder implementation in PyTorch.
    
    This class implements an autoencoder with sparsity constraints on the latent representation.
    The model includes methods for saving/loading, computing sparsity loss, and summarizing its configuration.
    
    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Dimensionality of the latent space.
        activation (str): Activation function to use in the encoder (default: "relu").
        sparsity_target (float): Target sparsity level for latent activations.
        sparsity_penalty (float): Weight for the sparsity regularization term.
        layer_index (int, optional): Index of the layer in a larger model (used for context).
    """

    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 activation: str = "relu", 
                 sparsity_target: float = 0.03, 
                 sparsity_penalty: float = 0.0001,
                 layer_index: int = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.decoder = torch.nn.Linear(hidden_dim, input_dim)
        
        # Validate and set activation function
        if activation not in ["relu", "tanh", "sigmoid", "leaky_relu"]:
            raise ValueError(f"Unsupported activation function: {activation}")
        self.activation_fn = getattr(torch.nn.functional, activation)
        
        self.sparsity_target = sparsity_target
        self.sparsity_penalty = sparsity_penalty
        self.layer_index = layer_index
        self.config = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "activation": self.activation_fn.__name__,
            "sparsity_target": self.sparsity_target,
            "sparsity_penalty": self.sparsity_penalty,
            "layer_index": self.layer_index
        }

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Perform the forward pass through the sparse autoencoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - recon (torch.Tensor): Reconstructed input of shape (batch_size, input_dim).
                - z (torch.Tensor): Latent representation of shape (batch_size, hidden_dim).
        """
        z = self.activation_fn(self.encoder(x))
        recon = self.decoder(z)
        return recon, z

    def compute_sparsity_loss(self, z: Tensor) -> Tensor:
        """
        Compute the sparsity loss using L1 regularization.
        
        Args:
            z (torch.Tensor): Latent representation of shape (batch_size, hidden_dim).
        
        Returns:
            torch.Tensor: Sparsity loss value.
        """
        return self.sparsity_penalty * torch.mean(torch.abs(z))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SparseAutoencoder":
        """
        Create a SparseAutoencoder instance from a configuration dictionary.
        
        Args:
            config (Dict[str, Any]): Dictionary containing model hyperparameters.
        
        Returns:
            SparseAutoencoder: Initialized model instance.
        """
        print(config)
        return cls(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            activation=config.get("activation", "relu"),
            sparsity_target=config.get("sparsity_target", 0.03),
            sparsity_penalty=config.get("sparsity_penalty", 0.0001),
            layer_index=config.get("layer_index", 12)
        )

    def save(self, path: Union[str, Path]):
        """
        Save the model's state_dict and configuration to the specified directory.
        
        Args:
            path (Union[str, Path]): Directory path where the model will be saved.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "pytorch_model.bin")
        with open(path / "config.yaml", "w") as f:
            yaml.dump(self.config, f)

    @classmethod
    def load(cls, path: Union[str, Path], device: str = None) -> "SparseAutoencoder":
        """
        Load a SparseAutoencoder instance from the specified directory.
        
        Args:
            path (Union[str, Path]): Directory path where the model is saved.
            device (str, optional): Device to load the model onto (e.g., "cpu" or "cuda").
        
        Returns:
            SparseAutoencoder: Loaded model instance.
        """
        path = Path(path)
        with open(path / "config.yaml", "r") as f:
            config = yaml.safe_load(f)
        model = cls.from_config(config)
        state_dict = torch.load(path / "pytorch_model.bin", map_location=device or "cpu")
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def compute_topk_activations_mean(hidden_states: Tensor, sae_model: "SparseAutoencoder", top_k: int = 20) -> float:
        """
        Compute the mean of the top-k activations in the latent space.
        
        Args:
            hidden_states (torch.Tensor): Hidden states from an intermediate layer of shape (1, seq_len, hidden_dim).
            sae_model (SparseAutoencoder): Trained SparseAutoencoder instance.
            top_k (int): Number of top activations to consider.
        
        Returns:
            float: Mean of the top-k activations.
        """
        _, z = sae_model(hidden_states.squeeze(0))  # shape: (seq_len, hidden_dim)
        token_scores = torch.abs(z).sum(dim=0)  # L1 norm per feature across tokens
        topk_vals, _ = torch.topk(token_scores, k=top_k)
        return topk_vals.mean().item()

    def summary(self, print_output: bool = True) -> Dict[str, Any]:
        """
        Generate a summary of the model's configuration and parameters.
        
        Args:
            print_output (bool): Whether to print the summary to the console.
        
        Returns:
            Dict[str, Any]: Summary information about the model.
        """
        info = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "activation": self.activation_fn.__name__,
            "sparsity_target": self.sparsity_target,
            "sparsity_penalty": self.sparsity_penalty,
            "layer_index": self.layer_index,
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

        if print_output:
            print("🔍 Sparse Autoencoder Summary:")
            for k, v in info.items():
                print(f"  {k}: {v}")

        return info