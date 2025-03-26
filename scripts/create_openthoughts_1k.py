from datasets import load_dataset, Dataset
import random
from huggingface_hub import login
import os

# Step 1: Load original dataset
ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train")

# Step 2: Sample 1000 examples
sampled = random.sample(list(ds), 1000)

# Step 3: Create new Dataset
small_ds = Dataset.from_list(sampled)

# Step 4: Push to HF Hub
# Replace with your token or use `huggingface-cli login` first

login(token=os.getenv("HF_TOKEN"))

dataset_name = "ernanhughes/openthoughts-1k"
small_ds.push_to_hub(dataset_name)
