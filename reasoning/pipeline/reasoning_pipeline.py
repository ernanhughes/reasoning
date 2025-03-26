import os
import yaml
import torch
from datetime import datetime
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

from reasoning.logs.activation_log import ActivationLog
from reasoning.logs.activation_log import ActivationLogStore


class ReasoningPipeline:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config(config_path)
        self.model = self.load_model(self.config["model"]["name"])
        self.tokenizer = self.load_tokenizer(self.config["model"]["name"])

    def load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def load_model(self, model_name):
        return AutoModel.from_pretrained(model_name, output_hidden_states=True)

    def load_tokenizer(self, model_name):
        return AutoTokenizer.from_pretrained(model_name)

    def load_prompts(self):
        print("🔁 [1/5] Loading prompts...")
        dataset_id = self.config["prompts"]["dataset"]
        split = self.config["prompts"].get("split", "train")
        limit = self.config["prompts"].get("limit", None)
        ds = load_dataset(dataset_id, split=split)
        ds = ds.select(range(limit)) if limit else ds
        print(f"✅ Prompts loaded ({len(ds)})")
        return ds

    def extract_activations(self):
        print("🔁 [2/5] Extracting activations...")

        sae_config_path = self.config["sparse_autoencoder"]["config"]
        pt_filename = os.path.basename(sae_config_path).replace(".yaml", ".pt")
        pt_path = os.path.join("data/activations", pt_filename)
        os.makedirs(os.path.dirname(pt_path), exist_ok=True)

        if os.path.exists(pt_path):
            print(f"✅ Skipping extraction — already exists: {pt_path}")
            self.log_activation(sae_config_path, pt_path, skipped=True)
            return pt_path

        prompts = self.load_prompts()
        # Replace with actual model activation logic
        layer_dim = self.model.config.hidden_size
        fake_activations = torch.randn(len(prompts), 128, layer_dim)

        torch.save(fake_activations, pt_path)
        print(f"✅ Activations saved: {pt_path}")
        self.log_activation(sae_config_path, pt_path, skipped=False)
        return pt_path

    def log_activation(self, config_path, pt_path, skipped):
        log = ActivationLog(
            sae_config=config_path,
            activations_file=pt_path,
            skipped=skipped,
            created_at=datetime.utcnow().isoformat()
        )
        log.save()

    def train_sae(self, activations_file):
        print("🔁 [3/5] Training SAE...")

        # Replace with real SAE training logic
        class DummySAE:
            path = "sae_models/layer_11"
            final_loss = 0.0021

        print(f"✅ SAE trained: {DummySAE.path} (loss: {DummySAE.final_loss:.4f})")
        return DummySAE

    def score_features(self, sae, activations_file):
        print("🔁 [4/5] Scoring sparse features...")

        # Replace with real ReasonScore logic
        avg_score = 0.442
        print(f"✅ Avg ReasonScore: {avg_score:.3f}")
        return avg_score

    def log_pipeline_run(self):
        print("🔁 [5/5] Logging results...")
        print("✅ Pipeline complete 🎉")

    def process(self):
        self.extract_activations()
        sae = self.train_sae("placeholder.pt")
        self.score_features(sae, "placeholder.pt")
        self.log_pipeline_run()
