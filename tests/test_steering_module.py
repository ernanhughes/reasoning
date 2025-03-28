import sys
import os

# Optional: Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dspy import Example

from transformers import AutoModel, AutoTokenizer

from reasoning.sae.model import SparseAutoencoder
from reasoning.steering.sae_steering_module import SAESteeringModule
from reasoning.scoring.reason_score import compute_mean_topk_feature_score

def run_test():
    # Load model + tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load trained SAE
    sae_path = "sae_models/tinyllama-tinyllama-1.1b-chat-v1.0_ernanhughes-openorca-1k-short_layer12"
    sae = SparseAutoencoder.load(sae_path)

    # Top reasoning features (from ReasonScore run)
    top_features = [3377, 2101, 4412]

    # Init DSPy steering module
    # steerer = SAESteeringModule(model, tokenizer, sae, top_features=top_features)
    steerer = SAESteeringModule(model_name, tokenizer, sae)

    # Test example
    example = Example(instruction="Explain why people vote in elections.")
    result = steerer(example)

    print("🧪 Steered Output:")
    print(result["output"])
    print("🔬 z_steered shape:", result["z_steered"].shape)
    # score = compute_mean_topk_feature_score(hidden, sae, top_k=20)
    # print(f"🧠 Mean Top-K Reasoning Activation: {score:.4f}")


if __name__ == "__main__":
    run_test()

