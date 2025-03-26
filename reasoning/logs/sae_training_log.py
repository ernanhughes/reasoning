import json
import os
from dataclasses import dataclass, asdict

@dataclass
class SAETrainingLog:
    sae_config: str
    activations_file: str
    final_loss: float
    epochs: int
    created_at: str

    def save(self, path="logs/sae_training_logs.jsonl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(asdict(self)) + "\n")
