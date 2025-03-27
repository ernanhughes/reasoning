from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os

@dataclass
class MemoryLog:
    step: str
    gpu_allocated_gb: float
    gpu_reserved_gb: float
    gpu_total_gb: float
    cpu_used_mb: float
    timestamp: str

    def save(self, log_path="logs/memory_log.jsonl"):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            json.dump(asdict(self), f)
            f.write("\n")
