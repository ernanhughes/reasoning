from dataclasses import dataclass
from reasoning.logs.base import Loggable

@dataclass
class MemoryLog(Loggable):
    step: str
    gpu_allocated_gb: float
    gpu_reserved_gb: float
    gpu_total_gb: float
    cpu_used_mb: float
    timestamp: str

    id: str = None
    created_at: str = None

    def get_log_path(self):
        return "logs/memory_vector_log.jsonl"
