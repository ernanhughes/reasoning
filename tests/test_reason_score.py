import torch
from transformers import AutoModel, AutoTokenizer
from reasoning.sae.model import SparseAutoencoder
from reasoning.scoring.reason_score import compute_mean_topk_feature_score

def test_reason_score():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # Load trained SAE
    sae_path = "sae_models/tinyllama-tinyllama-1.1b-chat-v1.0_ernanhughes-openorca-1k-short_layer12"
    sae = SparseAutoencoder.load(sae_path)

    # Tokenize and encode a prompt
    prompt = "Explain why people prefer solar energy."
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[sae.config["layer_index"]]  # [1, T, H]

    # Run score
    score = compute_mean_topk_feature_score(hidden, sae, top_k=20)
    print(f"🧠 Mean Top-K Reasoning Activation: {score:.4f}")
    assert isinstance(score, float) and score > 0, "ReasonScore should be a positive float"

if __name__ == "__main__":
    test_reason_score()
