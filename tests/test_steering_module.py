from reasoning.steering.sae_steering_module import SAESteeringModule

def test_sae_steering():
    model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    sae_path = "sae_models/tinyllama-tinyllama-1.1b-chat-v1.0_ernanhughes-openorca-1k-short_layer12"
    layer_index = 12
    top_k = 20

    steerer = SAESteeringModule(model, sae_path, layer_index, top_k=top_k)
    result = steerer("I believe the stock price dropped because of weak demand in Q4.")

    print(f"🧠 Reasoning Score: {result['score']}")
    assert "score" in result
    assert isinstance(result["score"], float)
