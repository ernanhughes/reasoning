import argparse
import os
import yaml
from transformers import AutoConfig
from datetime import datetime



def slugify(model_id):
    return model_id.lower().replace("/", "-").replace("_", "-")


def generate_sae_config(model_id, output_dir="configs/sae"):
    print(f"Fetching config for: {model_id}")
    config = AutoConfig.from_pretrained(model_id)

    hidden_size = getattr(config, "hidden_size", None)
    num_layers = getattr(config, "num_hidden_layers", None)
    model_type = getattr(config, "model_type", "unknown")

    if hidden_size is None or num_layers is None:
        raise ValueError("Model config missing hidden_size or num_hidden_layers.")

    suggested_layer = num_layers // 2
    model_slug = slugify(model_id)
    filename = f"{model_slug}-layer{suggested_layer}.yaml"
    full_path = os.path.join(output_dir, filename)

    header = (
        f"# Auto-generated SAE config\n"
        f"# Created by: get_reasoning_parameters.py\n"
        f"# Model: {model_id}\n"
        f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )

    sae_config = {
        "input_dim": hidden_size,
        "hidden_dim": hidden_size * 8,
        "activation": "relu",
        "sparsity_penalty": 0.0001,
        "sparsity_target": 0.03,
        "lr": 0.001,
        "batch_size": 64,
        "epochs": 10,
        "layer_index": suggested_layer,
        "train_size": 10000,
        "model_type": model_type
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(full_path, "w") as f:
        f.write(header)
        yaml.dump(sae_config, f)

    print(f"\n✅ Saved SAE config to: {full_path}\n")
    print("Model Info:")
    print(f"  Model Type:        {model_type}")
    print(f"  Hidden Size:       {hidden_size}")
    print(f"  Hidden Layers:     {num_layers}")
    print(f"  Suggested Layer:   {suggested_layer}")
    print("\nConfig Preview:")
    print(yaml.dump(sae_config, sort_keys=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SAE config from Hugging Face model.")
    parser.add_argument("--model", required=True, help="Hugging Face model ID")
    parser.add_argument("--output_dir", default="configs/sae", help="Where to save the config YAML")

    args = parser.parse_args()
    generate_sae_config(args.model, args.output_dir)
