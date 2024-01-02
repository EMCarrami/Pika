import argparse
import json
import subprocess
import time
from ast import literal_eval
from typing import Any, Dict

import wandb

from cprt.utils import ROOT


def gpu_usage_logger(wandb_config: Dict[str, str], gpu_id: int, log_interval: float = 0.1) -> None:
    """Log gpu usage every second to wandb."""
    wandb.init(**wandb_config, resume="allow")
    while True:
        gpu_usage = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader", f"--id={gpu_id}"]
        )
        wandb.log({"monitor/gpu_usage": int(gpu_usage.decode().strip().split()[0].strip())})
        time.sleep(log_interval)


def cli_parser() -> Dict[str, Any]:
    """Parse config file and update args."""
    parser = argparse.ArgumentParser()
    _, args = parser.parse_known_args()
    args_dict = {}
    for i in range(0, len(args), 2):
        k, v = args[i], args[i + 1]
        assert k.startswith("--"), f"invalid key {k}"
        assert not v.startswith("--"), f"invalid value {v}"
        k = k.replace("--", "")
        assert k not in args_dict, f"{k} key is used more than once."
        args_dict[k] = v
    try:
        config_path: str = args_dict.pop("config")
        config = load_config(config_path)
    except (KeyError, FileNotFoundError):
        raise ValueError("--config must be specified and direct to a valid config file.")
    for k, v in args_dict.items():
        keys = k.split(".")
        end_key = keys.pop()
        _config = config
        for _k in keys:
            _config = _config[_k]
        try:
            _config[end_key] = literal_eval(v)
        except ValueError:
            _config[end_key] = v
    return config


def load_config(path: str) -> Dict[str, Any]:
    """Load config file from path."""
    assert path.endswith(".json"), "only json config is supported."
    if path.startswith("/"):
        absolute_path = path
    elif "/" not in path:
        absolute_path = f"{ROOT}/configs/{path}"
    else:
        absolute_path = f"{ROOT}/{path}"
    with open(absolute_path, "r") as f:
        config: Dict[str, Any] = json.load(f)
    return config
