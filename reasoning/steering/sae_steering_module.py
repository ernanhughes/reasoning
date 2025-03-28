import torch
from dspy import Module
from transformers import AutoTokenizer, AutoModel
from reasoning.sae.model import SparseAutoencoder
from reasoning.sae.utils import preprocess_for_sae
from reasoning.scoring.reason_score import compute_mean_topk_feature_score

class SAESteeringModule(Module):
    def __init__(self, model_name: str, sae_path: str, layer_index: int, top_k: int = 20, sae_device: str = "cpu"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).eval()
        self.model.to(torch.device(sae_device))

        self.sae = SparseAutoencoder.load(sae_path)
        self.sae_device = torch.device(sae_device)
        self.layer_index = layer_index
        self.top_k = top_k

    def forward(self, prompt: str) -> dict:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden = outputs.hidden_states[self.layer_index]  # [1, T, H]

        sae_input = preprocess_for_sae(hidden, self.sae.config["input_dim"])

        # move to cpu
        self.sae.to(self.sae_device)
        sae_input = sae_input.to(self.sae_device)

        with torch.no_grad():
            _, z = self.sae(sae_input)

        score = compute_mean_topk_feature_score(hidden, self.sae, top_k=self.top_k)
        return {
            "prompt": prompt,
            "score": round(score, 5)
        }
