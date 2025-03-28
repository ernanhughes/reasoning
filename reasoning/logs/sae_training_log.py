from dataclasses import dataclass
from reasoning.logs.base import Loggable

@dataclass
class SAETrainingLog(Loggable):
    sae_config: str
    activations_file: str
    final_loss: float
    epochs: int
    model_summary: dict

    id: str = None
    created_at: str = None

    def get_log_path(self):
        return "logs/sae_training_log.jsonl"
