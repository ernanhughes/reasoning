import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datetime import datetime, timezone
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import math
from omegaconf import DictConfig, OmegaConf, MISSING
from dataclasses import dataclass
import logging

from reasoning.logs.activation_log import ActivationLog
from reasoning.logs.sae_training_log import SAETrainingLog
from reasoning.sae.model import SparseAutoencoder
from reasoning.logs.memory_tracker import log_memory
from reasoning.scoring.reason_score import (
    build_reasoning_mask,
    compute_reason_scores
)


@dataclass
class SAEConfig:
    input_dim: int = MISSING
    hidden_dim: int = MISSING
    activation: str = "relu"
    sparsity_target: float = 0.03
    sparsity_penalty: float = 0.0001
    lr: float = 0.001
    batch_size: int = 32
    epochs: int = 10


class ReasoningPipeline:
    """
    A pipeline for extracting activations, training a Sparse Autoencoder (SAE),
    scoring sparse features, and logging results.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.validate_config()
        self.model = self.load_model(self.config["model"]["name"])
        self.tokenizer = self.load_tokenizer(self.config["model"]["name"])
        self.batch_size = self.config.get("batch_size", 1)  # Default batch size = 1 for memory safety
        self.max_seq_len = self.config["model"].get("max_seq_len", 64)

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def validate_config(self):
        """Validate the configuration schema."""
        sae_config = OmegaConf.merge(OmegaConf.structured(SAEConfig), self.config["sparse_autoencoder"])
        self.config["sparse_autoencoder"] = sae_config

    def resolve_device(self, stage: str) -> torch.device:
        """Resolve the device for a specific pipeline stage."""
        if "pipeline" not in self.config or stage not in self.config["pipeline"]:
            logging.warning(f"Stage '{stage}' not found in config. Defaulting to 'cpu'.")
            return torch.device("cpu")
        return torch.device(self.config["pipeline"][stage].get("device", "cpu"))

    def load_model(self, model_name: str):
        """Load the pre-trained model."""
        return AutoModel.from_pretrained(model_name, output_hidden_states=True)

    def load_tokenizer(self, model_name: str):
        """Load the tokenizer for the pre-trained model."""
        return AutoTokenizer.from_pretrained(model_name)

    def load_prompts(self):
        """Load prompts from the dataset."""
        logging.info("Loading prompts...")
        dataset_id = self.config["prompts"]["dataset"]
        split = self.config["prompts"].get("split", "train")
        limit = self.config["prompts"].get("limit", None)
        ds = load_dataset(dataset_id, split=split)
        ds = ds.select(range(limit)) if limit else ds
        logging.info(f"Prompts loaded ({len(ds)})")
        return ds

    def extract_activations(self):
        """Extract activations from the model."""
        logging.info("Extracting activations...")
        sae_config_path = self.config["sparse_autoencoder"]["config"]
        pt_filename = os.path.basename(sae_config_path).replace(".yaml", ".pt")
        pt_path = os.path.join("data/activations", pt_filename)
        os.makedirs(os.path.dirname(pt_path), exist_ok=True)

        if os.path.exists(pt_path):
            logging.info(f"Skipping extraction — already exists: {pt_path}")
            self.log_activation(sae_config_path, pt_path, skipped=True)
            return pt_path

        prompts = self.load_prompts()
        activations = self.extract_layer_activations(prompts)
        torch.save(activations, pt_path)
        logging.info(f"Activations saved: {pt_path}")
        self.log_activation(sae_config_path, pt_path, skipped=False)
        return pt_path

    def log_activation(self, config_path: str, pt_path: str, skipped: bool):
        """Log activation extraction details."""
        status = "Skipped" if skipped else "Created"
        log = ActivationLog(
            sae_config=config_path,
            activations_file=pt_path,
            skipped=skipped,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        log.save()
        logging.info(f"Activation Log | Config: {config_path} | File: {pt_path} | Status: {status}")

    def extract_layer_activations(self, prompts):
        """Extract activations from a specific layer of the model."""
        logging.info("Extracting activations from model...")
        device = self.resolve_device("activation_extraction")
        self.model.to(device)
        self.model.eval()

        layer_index = self.config["model"]["layer_index"]
        pad_to_max = self.config["model"].get("pad_to_max", True)

        def collate_fn(batch):
            texts = [f"{ex['system_prompt'].strip()}\n{ex['question'].strip()}" for ex in batch]
            return self.tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length" if pad_to_max else "longest",
                truncation=True,
                max_length=self.max_seq_len
            )

        loader = DataLoader(prompts, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        all_activations = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting layer activations"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden = outputs.hidden_states[layer_index]  # [B, T, H]

                if hidden.shape[1] < self.max_seq_len:
                    pad_len = self.max_seq_len - hidden.shape[1]
                    hidden = F.pad(hidden, (0, 0, 0, pad_len), mode="constant", value=0)
                else:
                    hidden = hidden[:, :self.max_seq_len, :]

                all_activations.append(hidden.cpu())  # Offload from GPU immediately

        log_memory("After activation extraction")
        return torch.cat(all_activations, dim=0)

    def train_sae(self, activations_file: str):
        """Train the Sparse Autoencoder."""
        logging.info("Training SAE...")
        device = self.resolve_device("sae_training")

        sae_config_path = self.config["sparse_autoencoder"]["config"]
        with open(sae_config_path, "r") as f:
            sae_cfg = yaml.safe_load(f)

        activations = torch.load(activations_file)

        hidden_size = self.model.config.hidden_size
        sae_cfg["input_dim"] = self.max_seq_len * hidden_size
        logging.info(f"Auto-setting SAE input_dim = {sae_cfg['input_dim']} ({self.max_seq_len} x {hidden_size})")

        model = SparseAutoencoder.from_config(sae_cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=sae_cfg["lr"])
        loss_fn = torch.nn.MSELoss()
        batch_size = sae_cfg["batch_size"]
        epochs = sae_cfg["epochs"]

        model.train()
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch + 1}/{epochs}")
            epoch_start = time()

            num_batches = math.ceil(len(activations) / batch_size)
            logging.info(f"Epoch {epoch + 1}/{epochs} — {num_batches} batches")

            batch_bar = tqdm(
                range(0, len(activations), batch_size),
                desc=f"  Training Batches",
                leave=False
            )

            for step, i in enumerate(batch_bar):
                batch = activations[i:i+batch_size].to(device)
                batch = batch.view(batch.size(0), -1)

                expected_dim = sae_cfg["input_dim"]
                actual_dim = batch.shape[1]
                if actual_dim != expected_dim:
                    raise ValueError(f"Activation shape mismatch: got {actual_dim}, expected {expected_dim}")

                optimizer.zero_grad()
                recon, z = model(batch)
                loss = loss_fn(recon, batch) + model.sparsity_loss(z)
                loss.backward()
                optimizer.step()

                batch_bar.set_postfix(loss=loss.item())

            logging.info(f"Epoch {epoch + 1} complete (avg loss: {loss.item():.4f}) — Time: {time() - epoch_start:.1f}s")

        log_memory("After SAE training")

        output_dir = self.config["sparse_autoencoder"].get("path", "sae_models/layer_11")
        os.makedirs(output_dir, exist_ok=True)
        model.save(output_dir)

        logging.info(f"SAE trained: {output_dir} (loss: {loss.item():.4f})")
        log = SAETrainingLog(
            sae_config=sae_config_path,
            activations_file=activations_file,
            final_loss=loss.item(),
            epochs=epochs,
            created_at=datetime.now(timezone.utc).isoformat(),
            model_summary=model.summary(print_output=False)
        )
        log.save()

        return model

    def load_sae(self):
        """Load a trained Sparse Autoencoder."""
        output_dir = self.config["sparse_autoencoder"].get("path", "sae_models/layer_11")
        return SparseAutoencoder.load(output_dir)

    def score_features(self, sae, activations_file: str):
        """Score sparse features based on reasoning masks."""
        logging.info("Scoring sparse features...")

        device = self.resolve_device("feature_scoring")
        activations = torch.load(activations_file).to(device)  # [B, T, F]
        prompts = self.load_prompts()

        token_ids = self.tokenizer([p["question"] for p in prompts], return_tensors="pt", padding=True).input_ids

        rs_config = self.config.get("reason_score", {})
        mode = rs_config.get("mode", "keywords")
        reasoning_mask = build_reasoning_mask(token_ids, self.tokenizer, mode=mode, config=rs_config)

        reason_scores = compute_reason_scores(activations, reasoning_mask)
        avg_score = sum(reason_scores) / len(reason_scores)
        logging.info(f"Avg ReasonScore: {avg_score:.3f}")
        return reason_scores

    def log_pipeline_run(self):
        """Log the completion of the pipeline."""
        logging.info("Logging results...")
        logging.info("Pipeline complete 🎉")

    def process(self):
        """Run the entire pipeline."""
        activations_file = self.extract_activations()
        sae = self.train_sae(activations_file)
        self.score_features(sae, activations_file)
        self.log_pipeline_run()