import json
from typing import Any, Dict, Tuple

import torch
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

import wandb
from cprt.data.cprt_datamodule import CPrtDataModule
from cprt.model.cprt_model import CPrtModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_reduced_model(checkpoint_path: str) -> CPrtModel:
    """Load a reduced CPrtModel from checkpoint with weights for only trained parts."""
    nck = torch.load(checkpoint_path, map_location=device)
    model = CPrtModel(**nck["hyper_parameters"])
    missing = model.load_state_dict(nck["state_dict"], strict=False)
    assert len(missing.unexpected_keys) == 0
    return model


def load_from_checkpoint(
    config: Dict[str, Any], checkpoint_path: str, is_partial: bool = False
) -> Tuple[CPrtModel, CPrtDataModule]:
    """Load model and datamodule from checkpoint and config."""
    if is_partial:
        model = load_reduced_model(checkpoint_path)
    else:
        model = CPrtModel.load_from_checkpoint(checkpoint_path, map_location=device)
    config["model"] = dict(model.hparams)
    config["datamodule"]["language_model"] = model.hparams["language_model"]
    config["datamodule"]["protein_model"] = model.hparams["protein_model"]
    datamodule = CPrtDataModule(**config["datamodule"])
    return model, datamodule


if __name__ == "__main__":
    config_path = "configs/test_config.json"
    checkpoint = "model_checkpoints/epoch=0-step=224085.ckpt"
    with open(config_path, "r") as f:
        config: Dict[str, Any] = json.load(f)
    if "seed" in config:
        seed_everything(config["seed"])
    m, d = load_from_checkpoint(config, checkpoint, is_partial=False)
    wandb_logger = WandbLogger(**config["wandb"])
    trainer = Trainer(logger=wandb_logger, **config["trainer"])
    trainer.logger.log_hyperparams(config)
    trainer.test(m, d)
    wandb.finish()
