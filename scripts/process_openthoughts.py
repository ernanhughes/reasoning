import json
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import login

# === CONFIG ===
INPUT_PATH = "OpenThoughts-114k.jsonl"  # Make sure you download it first
HF_DATASET_NAME = "Opethat'snThoughts-Reasoning-Subset"
MIN_RESPONSE_TOKENS = 50  # Threshold for reasoning-rich responses

# === Step 1: Load raw JSONL ===
print("Loading data...")
with open(INPUT_PATH, "r") as f:
    samples = [json.loads(line) for line in f]

print(f"Total samples: {len(samples)}")

# === Step 2: Filter for reasoning-rich responses ===
def is_reasoning_rich(sample):
    return "response" in sample and len(sample["response"].split()) > MIN_RESPONSE_TOKENS

filtered = list(filter(is_reasoning_rich, samples))
print(f"Filtered to {len(filtered)} reasoning-rich examples.")

# === Step 3: Clean + unify schema ===
def clean(sample):
    return {
        "prompt": sample.get("question", sample.get("input", "")),
        "response": sample["response"]
    }

cleaned = list(map(clean, filtered))
dataset = Dataset.from_list(cleaned)

# === Step 4: Save locally ===
dataset.save_to_disk("openthoughts_filtered")
print("Saved to disk at './openthoughts_filtered'")

# === Step 5: Upload to Hugging Face ===
print("\nLog into Hugging Face CLI before continuing:")
print("Run: huggingface-cli login\n")

user_input = input("Continue with upload? (y/n): ")
if user_input.strip().lower() == "y":
    login()  # Interactive login (or use login(token=...) if scripted)
    dataset.push_to_hub(HF_DATASET_NAME)
    print(f"Uploaded dataset to: https://huggingface.co/datasets/<your-username>/{HF_DATASET_NAME}")
else:
    print("Upload canceled.")
