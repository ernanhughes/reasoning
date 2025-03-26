from datasets import load_dataset

dataset = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
print(dataset)

print(dataset["train"][0])