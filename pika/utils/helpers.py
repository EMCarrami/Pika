import argparse
import json
import os
import subprocess
import time
from ast import literal_eval
from typing import Any, Dict, Tuple

from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import WandbLogger
from loguru import logger

import wandb


class ExceptionHandlerCallback(Callback):
    """Callback to handle exceptions."""

    def on_exception(self, trainer: Trainer, model: LightningModule, exception: BaseException) -> None:
        """Run final logs."""
        model.on_train_epoch_end()
        # to trigger checkpoint upload
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.finalize("success")


def gpu_usage_logger(wandb_config: Dict[str, Any], gpu_id: int, log_interval: float = 0.1) -> None:
    """
    Log gpu usage every second to wandb.

    :param wandb_config: Wandb config to use. If supplied with id it will add the log to an existing run.
    :param gpu_id: id of the gpu to monitor. e.g. 0
    :param log_interval: log interval in seconds.
    """
    wandb.init(**wandb_config, resume="allow")
    while True:
        gpu_usage = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader", f"--id={gpu_id}"]
        )
        wandb.log({"monitor/gpu_usage": int(gpu_usage.decode().strip().split()[0].strip())})
        time.sleep(log_interval)


def cli_parser() -> Tuple[Dict[str, Any], str]:
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

    run_modes = ["train", "train_and_benchmark", "benchmark_only", "infer_only", "enquire", "train_classifier"]
    try:
        run_mode: str = args_dict.pop("run_mode")
        assert run_mode in run_modes
    except (KeyError, AssertionError):
        raise ValueError(f"--run_mode must be specified and be one of {run_modes}.")

    for k, v in args_dict.items():
        keys = k.split(".")
        end_key = keys.pop()
        _config = config
        for _k in keys:
            _config = _config[_k]
        try:
            _config[end_key] = literal_eval(v)
        except (ValueError, SyntaxError):
            _config[end_key] = v
    return config, run_mode


def load_config(path: str) -> Dict[str, Any]:
    """Load config file from path."""
    assert path.endswith(".json"), "only json config is supported."
    with open(path, "r") as f:
        config: Dict[str, Any] = json.load(f)
    return config


def file_path_assertions(file_path: str, exists_ok: bool, strict_extension: str | None = None) -> Tuple[str, str]:
    """
    Check validity of file_path and create parent dirs.

    :param file_path: file path to analyse/
    :param exists_ok: If False raises Exception when file exists. If True raises a warning.
    :param strict_extension: Whether to strictly check for a specific extension.

    :returns file name and file extension
    """
    base_name = os.path.basename(file_path)
    assert base_name, f"file path {file_path} should not point to a directory."
    assert "." in base_name, f"specify an extension or the file {file_path}"
    if strict_extension is not None:
        assert file_path.endswith(
            strict_extension.strip(".")
        ), f"file path must be a {strict_extension} file. {file_path} was given."

    if exists_ok:
        if os.path.isfile(file_path):
            logger.warning(f"{file_path} already exists. File will be overwritten, ensure this is intended.")
    else:
        assert not os.path.isfile(file_path), f"file {file_path} already present, provide a new file name."

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    fn, ext = os.path.splitext(base_name)
    return fn, ext
