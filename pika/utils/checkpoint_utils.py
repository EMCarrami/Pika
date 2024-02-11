from glob import glob
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
    if checkpoint_path.startswith("wandb:"):
        checkpoint_path = find_wandb_checkpoint(checkpoint_path.replace("wandb:", ""))
        logger.info(f"using wandb model checkpoint in {checkpoint_path}")

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
    """Download the model artifact into model_checkpoints and return checkpoint path."""
    api = wandb.Api()
    artifact = api.artifact(checkpoint, type="model")
    model_dir = checkpoint.split("/")[-1].replace(":", "_")
    artifact_dir: str = artifact.download(f"model_checkpoints/{model_dir}")
    ckpt_file = glob(f"{artifact_dir}/*.ckpt")
    assert len(ckpt_file) == 1, f"there are more than 1 ckpt file in {artifact_dir}: {ckpt_file}"
    return ckpt_file[0]


def load_reduced_model(checkpoint_path: str) -> PikaModel:
    """Load a reduced PikaModel from checkpoint with weights for only trained parts."""
    nck = torch.load(checkpoint_path, map_location=device)
    model = PikaModel(**nck["hyper_parameters"])
    missing = model.load_state_dict(nck["state_dict"], strict=False)
    assert len(missing.unexpected_keys) == 0
    return model
