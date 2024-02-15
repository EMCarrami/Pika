import argparse
import json
import os
import subprocess
import time
from ast import literal_eval
from typing import Any, Dict

from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from loguru import logger

import wandb


def gpu_usage_logger(wandb_config: Dict[str, Any], gpu_id: int, log_interval: float = 0.1) -> None:
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
    with open(path, "r") as f:
        config: Dict[str, Any] = json.load(f)
    return config


class ExceptionHandlerCallback(Callback):
    """Callback to handle exceptions."""

    def on_exception(self, trainer: Trainer, model: LightningModule, exception: BaseException) -> None:
        """Run final logs."""
        model.on_train_epoch_end()
        # to trigger checkpoint upload
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.finalize("success")


def get_output_file_path(config: Dict[str, Any]) -> str:
    """Get output file path and create directories."""
    if "save_file_path" in config:
        out_file = config["save_file_path"]
        if out_file == "auto":
            save_dir = "test_results"
            file_name = config["checkpoint"]["path"].split("/")[-1].split(".")[0]
            if "name" in config["wandb"]:
                file_name = f"{config['wandb']['name']}_{file_name.split('_')[-1]}"
            out_file = f"{save_dir}/{file_name}.tsv"
        else:
            assert out_file.endswith(".tsv"), "only csv format is supported"
            if len(out_file.split("/")) > 1:
                save_dir = "/".join(out_file.split("/")[:-1])
            else:
                save_dir = "test_results"
                out_file = f"{save_dir}/{out_file}"
        assert not os.path.isfile(out_file), f"{out_file} already exists"
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"predicted texts will be saved in {out_file}")
        assert isinstance(out_file, str)
        return out_file
    else:
        return ""
