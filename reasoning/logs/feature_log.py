from dataclasses import dataclass
from reasoning.logs.base import Loggable

@dataclass
class FeatureLog(Loggable):
    prompt_id: str               # Links back to PromptLog
    feature_id: int              # Feature index (0..F-1)
    reason_score: float          # Global ReasonScore
    topk_score: float            # Top-K normalized ReasonScore
    sae_config: str              # Path to SAE config file
    layer_index: int             # Optional: for sweep analysis

    id: str = None
    created_at: str = None

    def get_log_path(self):
        return "logs/feature_log.jsonl"
