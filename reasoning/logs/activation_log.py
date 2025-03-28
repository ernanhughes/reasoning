from dataclasses import dataclass, asdict
from datetime import datetime
import os
import json
from typing import List, Optional


LOG_PATH = "logs/activation_logs.jsonl"


@dataclass
class ActivationLog:
    sae_config: str
    activations_file: str
    skipped: bool

    id: str = None
    created_at: str = None

    def get_log_path(self):
        return "logs/activation_log.jsonl"



class ActivationLogStore:
    def __init__(self, path=LOG_PATH):
        self.path = path
        self._logs = self._load()

    def _load(self) -> List[ActivationLog]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r") as f:
            return [ActivationLog(**json.loads(line)) for line in f]

    def all(self) -> List[ActivationLog]:
        return self._logs

    def filter_by_config(self, config_name: str) -> List[ActivationLog]:
        return [log for log in self._logs if config_name in log.sae_config]

    def filter_by_layer(self, layer_index: int) -> List[ActivationLog]:
        keyword = f"layer{layer_index}"
        return [log for log in self._logs if keyword in log.sae_config]

    def latest_for(self, config_name: str) -> Optional[ActivationLog]:
        matches = self.filter_by_config(config_name)
        return sorted(matches, key=lambda x: x.created_at, reverse=True)[0] if matches else None
