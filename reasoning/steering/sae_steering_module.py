import torch
from dspy import Signature, Module
from transformers import AutoTokenizer, AutoModel
from reasoning.sae.model import SparseAutoencoder
from reasoning.sae.utils import preprocess_for_sae


class SAESteeringModule(Module):

    def __init__(self, model_name, sae: SparseAutoencoder, top_features=None, max_seq_len=64, device=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.sae = sae.to(self.device)
        self.max_seq_len = max_seq_len
        self.top_features = top_features or []  # List of feature IDs (int)

    def forward(self, example):
        prompt = example["instruction"]
        if not prompt:
            raise ValueError("Missing 'instruction' in input example.")

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_seq_len).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            layer_index = self.sae.layer_index
            hidden = outputs.hidden_states[layer_index]  # shape: [1, T, H]

            # Flatten for SAE: [1, T*H]
            sae_input = preprocess_for_sae(hidden, self.sae.config["input_dim"]).to(self.device)
            _, z = self.sae(sae_input)  # shape: [1, hidden_dim]

        # Steer activations by enhancing top features
        z_steered = z.clone()
        for f in self.top_features:
            if f < z_steered.shape[1]:
                z_steered[0, f] += 1.0  # Simple boosting

        z_steered = z_steered.detach().cpu().numpy()

        return {
            "output": prompt,  # (or use model.generate later)
            "z_steered": z_steered,
            "top_features": self.top_features
        }
