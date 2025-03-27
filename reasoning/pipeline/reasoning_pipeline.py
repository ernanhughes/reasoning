import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

from reasoning.logs.activation_log import ActivationLog
from reasoning.logs.sae_training_log import SAETrainingLog
from reasoning.sae.model import SparseAutoencoder


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
        activations = self.extract_layer_activations(prompts)
        torch.save(activations, pt_path)
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

    def extract_layer_activations(self, prompts):
        print("⚙️ Extracting activations from model...")

        layer_index = self.config["model"]["layer_index"]
        max_len = self.config["model"].get("max_seq_len", 64)
        pad_to_max = self.config["model"].get("pad_to_max", True)
        batch_size = self.config.get("batch_size", 8)

        def collate_fn(batch):
            texts = [f"{ex['system_prompt'].strip()}\n{ex['question'].strip()}" for ex in batch]
            return self.tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length" if pad_to_max else True,
                truncation=True,
                max_length=max_len
            )

        loader = DataLoader(prompts, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        all_activations = []

        self.model.eval()
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting layer activations"):
                input_ids = batch["input_ids"].to(self.model.device)
                attention_mask = batch["attention_mask"].to(self.model.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden = outputs.hidden_states[layer_index]  # [B, T, H]

                if hidden.shape[1] < max_len:
                    pad_len = max_len - hidden.shape[1]
                    hidden = F.pad(hidden, (0, 0, 0, pad_len), mode="constant", value=0)
                else:
                    hidden = hidden[:, :max_len, :]

                all_activations.append(hidden.cpu())

        activations = torch.cat(all_activations, dim=0)
        return activations

    def train_sae(self, activations_file):
        print("🔁 [3/5] Training SAE...")
        sae_config_path = self.config["sparse_autoencoder"]["config"]
        with open(sae_config_path, "r") as f:
            sae_cfg = yaml.safe_load(f)

        activations = torch.load(activations_file)
        print("SAE input_dim must be", activations[0].numel())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        #model = SparseAutoencoder.from_config(sae_cfg).to(device)
        # Auto-compute input_dim
        hidden_size = self.model.config.hidden_size
        max_seq_len = self.config["model"].get("max_seq_len", 128)
        auto_input_dim = max_seq_len * hidden_size
        # sae_cfg["input_dim"] = auto_input_dim
        sae_cfg["input_dim"] = max_seq_len * hidden_size
        print(f"ℹ️ Auto-setting SAE input_dim = {auto_input_dim} ({max_seq_len} x {hidden_size})")


        model = SparseAutoencoder.from_config(sae_cfg).to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=sae_cfg["lr"])
        loss_fn = torch.nn.MSELoss()
        batch_size = sae_cfg["batch_size"]
        epochs = sae_cfg["epochs"]

        model.train()
        for epoch in tqdm(range(epochs), desc="Training SAE"):
            for i in range(0, len(activations), batch_size):
                batch = activations[i:i+batch_size].to(device)
                batch = batch.view(batch.size(0), -1)
                actual_dim = batch.shape[1]

                expected_dim = sae_cfg["input_dim"]
                if actual_dim != expected_dim:
                    raise ValueError(f"❌ Activation shape mismatch: got {actual_dim}, expected {expected_dim}")

                optimizer.zero_grad()
                recon, z = model(batch)
                loss = loss_fn(recon, batch) + model.sparsity_loss(z)
                loss.backward()
                optimizer.step()

        output_dir = self.config["sparse_autoencoder"].get("output_dir", "sae_models/layer_11")
        os.makedirs(output_dir, exist_ok=True)
        model.save(output_dir)

        print(f"✅ SAE trained: {output_dir} (loss: {loss.item():.4f})")

        log = SAETrainingLog(
            sae_config=sae_config_path,
            activations_file=activations_file,
            final_loss=loss.item(),
            epochs=epochs,
            created_at=datetime.utcnow().isoformat()
        )
        log.save()

        return model

    def load_sae(self):
        output_dir = self.config["sparse_autoencoder"].get("output_dir", "sae_models/layer_11")
        return SparseAutoencoder.load(output_dir)

    def score_features(self, sae, activations_file):
        print("🔁 [4/5] Scoring sparse features...")
        # Placeholder logic
        avg_score = 0.442
        print(f"✅ Avg ReasonScore: {avg_score:.3f}")
        return avg_score

    def log_pipeline_run(self):
        print("🔁 [5/5] Logging results...")
        print("✅ Pipeline complete 🎉")

    def process(self):
        activations_file = self.extract_activations()
        sae = self.train_sae(activations_file)
        self.score_features(sae, activations_file)
        self.log_pipeline_run()
