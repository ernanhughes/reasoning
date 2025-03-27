from datasets import load_dataset, Dataset
import random

# Load the full dataset
ds = load_dataset("Open-Orca/OpenOrca", split="train")

# Define your filtering logic (by character length or token length)
def is_short(example):
    combined = f"{example['system_prompt']}\n{example['question']}"
    return len(combined) < 300

# Apply the filter and shuffle
filtered = ds.filter(is_short)
filtered = filtered.shuffle(seed=42).select(range(1000))  # keep first 1000

# Save locally or push to HF
filtered.push_to_hub("ernanhughes/openorca-1k-short")
