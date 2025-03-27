# scripts/generate_sae_config.py

import os
import yaml
from datetime import datetime
from transformers import AutoModel

def generate_sae_config(pipeline_config_path, output_path):
    with open(pipeline_config_path, "r") as f:
        pipeline_cfg = yaml.safe_load(f)

    model_name = pipeline_cfg["model"]["name"]
    layer_index = pipeline_cfg["model"]["layer_index"]
    max_seq_len = pipeline_cfg["model"].get("max_seq_len", 128)
    batch_size = pipeline_cfg.get("batch_size", 8)

    # Load model to get hidden size
    model = AutoModel.from_pretrained(model_name)
    hidden_size = model.config.hidden_size
    input_dim = max_seq_len * hidden_size

    sae_config = {
        "activation": "relu",
        "batch_size": batch_size,
        "epochs": 10,
        "hidden_dim": input_dim // 8,
        "input_dim": input_dim,
        "layer_index": layer_index,
        "lr": 0.001,
        "model_type": "llama",
        "sparsity_penalty": 0.0001,
        "sparsity_target": 0.03,
        "train_size": 1000,
    }

    # Write YAML with a nice comment
    comment = f"""# Auto-generated SAE config
# Created by: generate_sae_config.py
# Model: {model_name}
# Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}

"""

    with open(output_path, "w") as f:
        f.write(comment)
        yaml.dump(sae_config, f)

    print(f"✅ SAE config generated at: {output_path}")
    print(f"ℹ️ input_dim = {input_dim}, hidden_size = {hidden_size}, max_seq_len = {max_seq_len}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", required=True, help="Path to pipeline_config.yaml")
    parser.add_argument("--output", required=True, help="Output SAE config path")
    args = parser.parse_args()

    generate_sae_config(args.pipeline, args.output)
