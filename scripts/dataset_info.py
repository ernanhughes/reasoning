from datasets import load_dataset

dataset = load_dataset("open-thoughts/OpenThoughts-114k", split="train")
print(dataset)
print(dataset.column_names)

ds = load_dataset("Open-Orca/OpenOrca", split="train")
print(ds.column_names)
# ['id', 'prompt', 'response']
