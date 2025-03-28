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
    model_summary: dict

    def save(self, path="logs/sae_training_log.jsonl"):
        record = asdict(self)
        # Serialize dict as JSON string if needed
        record["model_summary"] = json.dumps(record["model_summary"])
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")