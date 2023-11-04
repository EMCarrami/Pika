import subprocess
import time
from typing import Dict

import wandb


def gpu_usage_logger(wandb_config: Dict[str, str], gpu_id: int, log_interval: float = 0.1) -> None:
    """Log gpu usage every second to wandb."""
    wandb.init(**wandb_config, resume="allow")
    while True:
        gpu_usage = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader", f"--id={gpu_id}"]
        )
        wandb.log({"monitor/gpu_usage": int(gpu_usage.decode().strip().split()[0].strip())})
        time.sleep(log_interval)
