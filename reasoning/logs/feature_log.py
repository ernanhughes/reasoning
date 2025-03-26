import json
import os
from dataclasses import dataclass, asdict

@dataclass
class FeatureLog:
    sae_config: str
    layer_index: int
    top_features: list[int]
    avg_reason_score: float
    reason_score_type: str
    created_at: str

    def save(self, path="logs/feature_logs.jsonl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(asdict(self)) + "\n")
