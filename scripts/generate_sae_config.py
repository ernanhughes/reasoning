# scripts\generate_sae_config.py
import os
import yaml
import argparse
from datetime import datetime
from transformers import AutoModel


def slugify(name: str) -> str:
    return name.lower().replace("/", "-").replace("_", "-")


def generate_configs(model_name: str, dataset: str, layer_index: int, max_seq_len: int,
                     sae_dir="configs/sae", pipeline_dir="configs/pipeline"):

    # --- Load model to get input dims ---
    model = AutoModel.from_pretrained(model_name)
    hidden_size = model.config.hidden_size
    input_dim = max_seq_len * hidden_size
    hidden_dim = input_dim // 8

    # --- Naming conventions ---
    model_slug = slugify(model_name)
    dataset_slug = slugify(dataset)
    base_name = f"{model_slug}_{dataset_slug}_layer{layer_index}"
    sae_path = os.path.join(sae_dir, f"{base_name}.yaml")
    pipeline_path = os.path.join(pipeline_dir, f"{base_name}.yaml")

    os.makedirs(sae_dir, exist_ok=True)
    os.makedirs(pipeline_dir, exist_ok=True)

    # --- SAE Config ---
    sae_config = {
        "activation": "relu",
        "batch_size": 1,
        "epochs": 10,
        "hidden_dim": hidden_dim,
        "input_dim": input_dim,
        "layer_index": layer_index,
        "lr": 0.001,
        "model_type": "llama",
        "sparsity_penalty": 0.0001,
        "sparsity_target": 0.03,
        "train_size": 1000,
        "_generated": {
            "by": "generate_sae_config.py",
            "model": model_name,
            "dataset": dataset,
            "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    # --- Pipeline Config ---
    pipeline_config = {
        "batch_size": 1,
        "model": {
            "name": model_name,
            "layer_index": layer_index,
            "max_seq_len": max_seq_len,
            "pad_to_max": True
        },
        "prompts": {
            "dataset": dataset,
            "split": "train",
            "limit": 1000,
            "text_column": "question"
        },
        "sparse_autoencoder": {
            "path": f"sae_models/{base_name}",
            "train_if_missing": True,
            "config": sae_path
        },
        "reason_score": {
            "mode": "keywords",
            "keyword_file": "configs/reasoning_keywords.txt",
            "token_file": "configs/reasoning_token_ids.txt",
            "top_k_normalization": {
                "enabled": False,    # ← Default OFF
                "k": 10
            },
        },
        "database": {
            "uri": "postgresql://reasoning:reasoning@localhost:5432/reasoning"
        },
        "pipeline": {
            "tokenization": {
                "enabled": True,
                "enabled_reason": "Always needed.",
                "device": "cpu",
                "reason": "Tokenization is lightweight and runs well on CPU."
            },
            "activation_extraction": {
                "enabled": True,
                "enabled_reason": "Required to get activations.",
                "device": "cuda",
                "reason": "Model forward pass benefits from GPU."
            },
            "activation_storage": {
                "enabled": True,
                "enabled_reason": "We want to persist extracted data.",
                "device": "cpu",
                "reason": "Storage can run on CPU safely."
            },
            "sae_training": {
                "enabled": True,
                "enabled_reason": "Train a new SAE for this setup.",
                "device": "cpu",
                "reason": "Efficient on CPU unless input_dim is huge."
            },
            "sae_steering": {
                "enabled": True,
                "enabled_reason": "SAE steering.",
                "device": "cpu",
                "reason": "To much memory to run on GPU."
            },
            "feature_scoring": {
                "enabled": True,
                "enabled_reason": "Compute ReasonScore for activations.",
                "device": "cpu",
                "reason": "Sparse vector scoring is CPU-safe."
            }
        },
        "_generated": {
            "by": "generate_sae_config.py",
            "model": model_name,
            "dataset": dataset,
            "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    # --- Save files ---
    with open(sae_path, "w") as f:
        f.write(f"# Auto-generated SAE config\n")
        yaml.dump(sae_config, f)

    with open(pipeline_path, "w") as f:
        f.write(f"# Auto-generated pipeline config\n")
        yaml.dump(pipeline_config, f)

    print(f"✅ SAE config saved:      {sae_path}")
    print(f"✅ Pipeline config saved: {pipeline_path}")
    print(f"ℹ️ input_dim = {input_dim}, hidden_size = {hidden_size}, max_seq_len = {max_seq_len}")

    # --- Update Hydra entrypoint config.yaml ---
    if not args.no_hydra_update:
        hydra_config_path = os.path.join("configs", "config.yaml")
        hydra_entry = {
            "defaults": [
                {"pipeline": os.path.splitext(os.path.basename(pipeline_path))[0]}
            ]
        }

        with open(hydra_config_path, "w") as f:
            yaml.dump(hydra_entry, f)

        print(f"🔄 Hydra default updated in: {hydra_config_path}")
    else:
        print("⏭️ Skipped Hydra config update (--no-hydra-update)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model ID")
    parser.add_argument("--dataset", required=True, help="Dataset name (Hugging Face)")
    parser.add_argument("--layer", type=int, default=12, help="Layer index to extract from")
    parser.add_argument("--max-seq-len", type=int, default=64, help="Maximum token length")
    parser.add_argument("--sae-dir", default="configs/sae", help="Directory to save SAE config")
    parser.add_argument("--pipeline-dir", default="configs/pipeline", help="Directory to save pipeline config")
    parser.add_argument("--no-hydra-update", action="store_true", help="Do not update Hydra's configs/config.yaml default pipeline")

    args = parser.parse_args()

    generate_configs(
        model_name=args.model,
        dataset=args.dataset,
        layer_index=args.layer,
        max_seq_len=args.max_seq_len,
        sae_dir=args.sae_dir,
        pipeline_dir=args.pipeline_dir,
    )

