from typing import Any, Dict

import wandb
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from transformers import logging as transformers_logging

from cprt.data.classification_datamodule import ClassificationDataModule
from cprt.model.classification_baseline import ProteinClassificationModel
from cprt.utils.helpers import cli_parser

transformers_logging.set_verbosity_error()


def train_classifier(config: Dict[str, Any], log_to_wandb: bool = False) -> None:
    """Run protein classifier training."""
    if "seed" in config:
        seed_everything(config["seed"])

    config["datamodule"]["protein_model"] = config["model"]["protein_model"]
    datamodule = ClassificationDataModule(**config["datamodule"])

    model = ProteinClassificationModel(**config["model"], num_classes=datamodule.num_classes)

    group_name = (
        f"{config['model']['classifier']}_{config['model']['protein_model']}_"
        f"{config['datamodule']['classification_task']}"
    )

    if log_to_wandb:
        config["wandb"]["group"] = group_name
        wandb_logger = WandbLogger(**config["wandb"])
        trainer = Trainer(logger=wandb_logger, **config["trainer"])
        trainer.logger.log_hyperparams(config)
        trainer.fit(model, datamodule)
        wandb.finish()
    else:
        trainer = Trainer(**config["trainer"])
        trainer.fit(model, datamodule)


if __name__ == "__main__":
    config = cli_parser()
    train_classifier(config, log_to_wandb=config.pop("log_to_wandb", False))
