# extract_activations_tinyllama.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import os
import json

# scripts/extract_activations.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import numpy as np
import os
from tqdm import tqdm

def extract_activations(model_name, layer_index, dataset_path, max_samples, save_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.eval().cuda()

    ds = load_from_disk(dataset_path)
    activations = []

    for sample in tqdm(ds.select(range(min(len(ds), max_samples)))):
        tokens = tokenizer(sample["prompt"], return_tensors="pt", truncation=True).to("cuda")
        with torch.no_grad():
            out = model(**tokens)
            hidden = out.hidden_states[layer_index].squeeze(0)  # shape: (seq_len, dim)
            activations.append(hidden.cpu().numpy())

    np.save(save_path, np.concatenate(activations, axis=0))  # shape: (N, d_model)
    print(f"Saved activations: {save_path}")


# Config
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LAYER_INDEX = 12  # Choose a layer index
OUTPUT_DIR = "activations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, output_hidden_states=True, torch_dtype=torch.float16
)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Dataset: You can replace this with any file or prompt list
sample_texts = [
    "Let me think step by step.",
    "To solve this, I will consider both options.",
    "Hmm, that seems tricky. Let's break it down.",
    "I'll start with the first principle.",
    "This is a math problem. Let's analyze it logically."
]

all_activations = []

for text in tqdm(sample_texts, desc="Extracting activations"):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.hidden_states[LAYER_INDEX]  # shape: (1, seq_len, dim)
    hidden_np = hidden.squeeze(0).cpu().numpy()  # shape: (seq_len, dim)
    all_activations.append({
        "text": text,
        "activations": hidden_np.tolist()
    })

# Save output
with open(os.path.join(OUTPUT_DIR, "tinyllama_activations.json"), "w") as f:
    json.dump(all_activations, f)

print("Activations saved!")
