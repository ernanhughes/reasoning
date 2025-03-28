import os
import torch
from datetime import datetime, timezone
from transformers import AutoTokenizer, AutoModel

from reasoning.logs.prompt_log import PromptLog
from reasoning.logs.feature_vector_log import FeatureVectorLog
from reasoning.logs.base import Loggable
from reasoning.scoring.reason_score import (
    build_reasoning_mask,
    compute_reason_scores,
    compute_topk_reason_scores,
)
from reasoning.sae.model import SparseAutoencoder
from reasoning.sae.utils import preprocess_for_sae


def test_score_and_log_prompt():
    # ✅ Step 1: Define prompt and model
    prompt_text = "I think the reason we saw a decline is due to market pressure."
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    # ✅ Step 2: Tokenize and log the prompt
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    tokens = inputs["input_ids"][0].tolist()

    prompt_id = PromptLog.generate_id()
    created_at = datetime.now(timezone.utc).isoformat()

    prompt_log = PromptLog(
        prompt_id=prompt_id,
        model=model_name,
        dataset="test_dataset",
        text=prompt_text,
        tokens=tokens,
        created_at=created_at
    )
    prompt_log.save()
    print(f"✅ Prompt logged: {prompt_id}")

    # ✅ Step 3: Load trained SAE and extract hidden layer
    sae_path = "sae_models/tinyllama-tinyllama-1.1b-chat-v1.0_ernanhughes-openorca-1k-short_layer12"
    sae = SparseAutoencoder.load(sae_path)
    layer_index = sae.config["layer_index"]
    expected_dim = sae.config["input_dim"]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[layer_index]  # [1, T, H]
        sae_input = preprocess_for_sae(hidden, expected_dim)
        _, z = sae(sae_input)
        z = z.view(1, hidden.shape[0], -1)  # [1, T, F]

        # ✅ Step 4: Build reasoning mask and compute ReasonScores
        reasoning_mask = build_reasoning_mask(inputs["input_ids"], tokenizer)
        reason_scores = compute_reason_scores(z, reasoning_mask)
        topk_scores = compute_topk_reason_scores(z, reasoning_mask, k=10)

    print(f"🧠 Total features scored: {len(reason_scores)}")
    print(f"🧪 First ReasonScores: {reason_scores[:5]}")
    print(f"📈 First TopK Scores: {topk_scores[:5]}")

    # ✅ Step 5: Log feature scores
    sae_config_path = os.path.join(sae_path, "config.yaml")
    log = FeatureVectorLog(
        prompt_id=prompt_id,
        reason_scores=reason_scores,
        topk_scores=topk_scores,
        sae_config=sae_config_path,
        layer_index=layer_index,
        created_at=created_at
    )
    log.save()

    print(f"✅ Feature scores logged for {len(reason_scores)} features.")


if __name__ == "__main__":
    Loggable.use_db = True
    test_score_and_log_prompt()
