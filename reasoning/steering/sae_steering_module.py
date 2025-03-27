import torch
import numpy as np
from dspy.primitives.program import Module


class SAESteeringModule(Module):
    def __init__(self, model, tokenizer, sae, top_feature_ids: list[int], injection_value: float = 3.0):
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae
        self.top_feature_ids = top_feature_ids
        self.injection_value = injection_value

    def forward(self, example: dict, trace: dict = None) -> dict:
        # Prepare input text
        prompt = example["instruction"]
        if not prompt:
            raise ValueError("Missing 'instruction' in input example.")
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]

        # Get hidden states from model
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # You may want to use specific layer

        # Flatten and encode with SAE
        sae_input = hidden.view(hidden.size(0), -1)

        # Get expected input size from SAE
        expected_input_dim = self.sae.input_dim
        actual_input_dim = sae_input.shape[1]

        if actual_input_dim < expected_input_dim:
            pad_len = expected_input_dim - actual_input_dim
            sae_input = torch.nn.functional.pad(sae_input, (0, pad_len))
        elif actual_input_dim > expected_input_dim:
            sae_input = sae_input[:, :expected_input_dim]

        _, z = self.sae(sae_input)

        # Inject top features
        z_steered = z.clone()
        for fid in self.top_feature_ids:
            z_steered[:, fid] = self.injection_value

        # Decode (this is stub — real integration needed)
        # For now, return the steered z vector as the output trace
        return {
            "output": prompt + " (steered)",
            "z_steered": z_steered.detach().cpu().numpy()
        }

    @staticmethod
    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """
        Safely convert a tensor to NumPy, detaching from the graph and moving to CPU.
        """
        return tensor.detach().cpu().numpy()
