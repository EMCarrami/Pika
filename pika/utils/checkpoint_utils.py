import json
import os
from typing import Any, Dict, Tuple

import torch
import wandb
from lightning import seed_everything
from loguru import logger

from pika.datamodule.pika_datamodule import PikaDataModule
from pika.model.pika_model import PikaModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_from_checkpoint(
    config: Dict[str, Any], checkpoint_path: str, is_partial: bool = False
) -> Tuple[PikaModel, PikaDataModule]:
    """Load model and datamodule from checkpoint and config."""
    if os.path.isfile("model_checkpoints/model_checkpoints.json"):
        with open("model_checkpoints/model_checkpoints.json", "r") as file:
            checkpoints = json.load(file)

    if checkpoint_path in checkpoints["path"]:
        artifact_dir = find_wandb_checkpoint(checkpoints["path"][checkpoint_path])
        config["seed"] = checkpoints["seed"][checkpoint_path]
        logger.info(f"seed was automatically set to {config['seed']} for wandb model {checkpoint_path}")
        checkpoint_path = f"{artifact_dir}/{checkpoint_path}"

    if is_partial:
        model = load_reduced_model(checkpoint_path)
    else:
        model = PikaModel.load_from_checkpoint(checkpoint_path, map_location=device)
    config["model"] = dict(model.hparams)
    config["datamodule"]["language_model"] = model.hparams["language_model"]
    config["datamodule"]["protein_model"] = model.hparams["protein_model"]
    if "seed" in config:
        seed_everything(config["seed"])
    datamodule = PikaDataModule(**config["datamodule"])
    return model, datamodule


def find_wandb_checkpoint(checkpoint: str) -> str:
    """Download the model artifact into model_checkpoints."""
    api = wandb.Api()
    artifact = api.artifact(checkpoint, type="model")  # type: ignore[no-untyped-call]
    artifact_dir: str = artifact.download("model_checkpoints")
    return artifact_dir


def load_reduced_model(checkpoint_path: str) -> PikaModel:
    """Load a reduced PikaModel from checkpoint with weights for only trained parts."""
    nck = torch.load(checkpoint_path, map_location=device)
    model = PikaModel(**nck["hyper_parameters"])
    missing = model.load_state_dict(nck["state_dict"], strict=False)
    assert len(missing.unexpected_keys) == 0
    return model
