import os
import psutil
import torch
from datetime import datetime
from reasoning.logs.memory_log import MemoryLog

def log_memory(step: str, log_path="logs/memory_log.jsonl", verbose=True):
    if not torch.cuda.is_available():
        return

    gpu_allocated = torch.cuda.memory_allocated() / 1024**3
    gpu_reserved = torch.cuda.memory_reserved() / 1024**3
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    process = psutil.Process(os.getpid())
    cpu_used = process.memory_info().rss / 1024**2

    log = MemoryLog(
        step=step,
        gpu_allocated_gb=round(gpu_allocated, 2),
        gpu_reserved_gb=round(gpu_reserved, 2),
        gpu_total_gb=round(gpu_total, 2),
        cpu_used_mb=round(cpu_used, 2),
        timestamp=datetime.utcnow().isoformat()
    )

    log.save(log_path)

    if verbose:
        print(f"📊 [{step}] GPU: {log.gpu_allocated_gb:.2f} GB allocated, "
              f"{log.gpu_reserved_gb:.2f} GB reserved, {log.gpu_total_gb:.2f} GB total | "
              f"CPU: {log.cpu_used_mb:.2f} MB used")
