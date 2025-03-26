import torch
import torch.nn as nn
import numpy as np

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, expansion_factor=16):
        super().__init__()
        hidden_dim = input_dim * expansion_factor
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        h = self.encoder(x)
        x_hat = self.decoder(h)
        return x_hat, h

def train_sae(activations_path, save_path, beta=5.0, epochs=10, batch_size=64):
    X = torch.tensor(np.load(activations_path), dtype=torch.float32)
    dim = X.shape[1]
    model = SparseAutoencoder(input_dim=dim).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        perm = torch.randperm(X.size(0))
        X = X[perm]

        for i in range(0, X.size(0), batch_size):
            batch = X[i:i+batch_size].cuda()
            x_hat, h = model(batch)
            recon = loss_fn(x_hat, batch)
            sparsity = h.abs().sum() / h.numel()
            loss = recon + beta * sparsity

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Sparsity={sparsity.item():.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"SAE saved to {save_path}")
