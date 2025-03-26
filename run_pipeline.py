# run_pipeline.py

import yaml
import os

from scripts.extract_activations import extract_activations
from scripts.train_sae import train_sae

def main():
    with open("scripts/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    print("=== Stage 1: Extracting Activations ===")
    os.makedirs(os.path.dirname(cfg["activations_path"]), exist_ok=True)
    extract_activations(
        model_name=cfg["model_name"],
        layer_index=cfg["layer_index"],
        dataset_path=cfg["dataset_path"],
        max_samples=cfg["max_samples"],
        save_path=cfg["activations_path"]
    )

    print("\n=== Stage 2: Training Sparse Autoencoder ===")
    os.makedirs(os.path.dirname(cfg["sae_model_path"]), exist_ok=True)
    train_sae(
        activations_path=cfg["activations_path"],
        save_path=cfg["sae_model_path"],
        beta=cfg.get("beta", 5.0),
        epochs=cfg.get("epochs", 10),
        batch_size=cfg.get("batch_size", 64)
    )

if __name__ == "__main__":
    main()
