import subprocess
import time
from typing import Any, Dict

import wandb


def gpu_usage_logger(wandb_config: Dict[str, str], stop_event: Any, gpu_id: int, log_interval: int = 1) -> None:
    """Log gpu usage every second to wandb."""
    wandb.init(**wandb_config, resume="allow")
    while not stop_event.is_set():
        gpu_usage = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader", f"--id={gpu_id}"]
        )
        wandb.log({"gpu_usage": int(gpu_usage.decode().strip())})
        time.sleep(log_interval)
