import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import argparse
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def load_config(path):
    with open(path) as f:
        return json.load(f)

def save_model(model, output_dir, config):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

def log_training(output_dir, layer_index, loss):
    log = {
        "timestamp": datetime.now().isoformat(),
        "layer_index": layer_index,
        "final_loss": loss
    }
    with open(os.path.join(output_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

def train_sae(activations, config, output_dir):
    input_dim = config["input_dim"]
    hidden_dim = config["hidden_dim"]
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 64)
    lr = config.get("lr", 1e-3)
    sparsity_penalty = config.get("sparsity_penalty", 1e-4)

    model = SparseAutoencoder(input_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    data = torch.tensor(activations, dtype=torch.float32)
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch, in loader:
            optimizer.zero_grad()
            x_hat, z = model(batch)
            mse_loss = criterion(x_hat, batch)
            sparse_loss = sparsity_penalty * torch.mean(torch.abs(z))
            loss = mse_loss + sparse_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    save_model(model, output_dir, config)
    log_training(output_dir, config.get("layer_index", -1), total_loss / len(loader))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to SAE config JSON")
    parser.add_argument("--activations", required=True, help="Path to .pt file with layer activations")
    parser.add_argument("--output_dir", required=True, help="Where to save the SAE model")
    args = parser.parse_args()

    config = load_config(args.config)
    activations = torch.load(args.activations)
    train_sae(activations, config, args.output_dir)
