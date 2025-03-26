import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from reasoning.sae import load_sae_model, compute_reason_score_for_layer

# Configuration
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat"
SAE_DIR = "./sae_models"
LAYER_RANGE = range(0, 32, 2)
PROMPT_SAMPLE = [
    "Let's think this through carefully.",
    "Because the supply chain was disrupted, earnings declined.",
    "To calculate revenue, multiply units sold by price per unit."
]

# Load model + tokenizer
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
model.eval()

layer_scores = []

for layer in tqdm(LAYER_RANGE, desc="Scanning layers"):
    all_scores = []

    for prompt in PROMPT_SAMPLE:
        tokens = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model(**tokens)
            hidden = out.hidden_states[layer]  # (1, seq_len, dim)

        # Load trained SAE for this layer
        sae_path = f"{SAE_DIR}/layer_{layer}"
        try:
            sae = load_sae_model(sae_path)
            reason_score = compute_reason_score_for_layer(hidden, sae)
            all_scores.append(reason_score)
        except FileNotFoundError:
            print(f"Skipping layer {layer} (no SAE found)")
            continue

    if all_scores:
        avg_score = np.mean(all_scores)
        layer_scores.append((layer, avg_score))

# Plot
layers, scores = zip(*layer_scores)
plt.figure(figsize=(8, 4))
plt.plot(layers, scores, marker="o")
plt.title("Average ReasonScore per Layer")
plt.xlabel("Layer Index")
plt.ylabel("Avg ReasonScore")
plt.grid(True)
plt.tight_layout()
plt.savefig("reason_score_by_layer.png")
print("✅ Saved plot to reason_score_by_layer.png")
