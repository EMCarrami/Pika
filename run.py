from datetime import datetime
from typing import Any, Dict

import wandb
from lightning import Callback, Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from transformers import logging as transformers_logging

from cprt.data.cprt_datamodule import CPrtDataModule
from cprt.model.cprt_model import CPrtModel
from cprt.utils import ROOT
from cprt.utils.helpers import cli_parser

transformers_logging.set_verbosity_error()


class ExceptionHandlerCallback(Callback):  # type: ignore[misc]
    """Callback to handle exceptions."""

    def on_exception(self, trainer: Trainer, model: CPrtModel, exception: Exception) -> None:
        """Run final logs."""
        model.on_train_epoch_end()


def train_cprt(config: Dict[str, Any], log_to_wandb: bool = False) -> None:
    """Run Cprt training."""
    if "seed" in config:
        seed_everything(config["seed"])

    config["datamodule"]["language_model"] = config["model"]["language_model"]
    config["datamodule"]["protein_model"] = config["model"]["protein_model"]

    if "[seed]" in config["datamodule"]["data_dict_path"]:
        config["datamodule"]["data_dict_path"] = config["datamodule"]["data_dict_path"].replace(
            "[seed]", str(config["seed"])
        )
    datamodule = CPrtDataModule(**config["datamodule"])

    multimodal_strategy = config["model"]["multimodal_strategy"]
    model = CPrtModel(**config["model"])

    if "n_vals_per_epoch" in config["trainer"]:
        n_vals = config["trainer"].pop("n_vals_per_epoch")
        config["trainer"]["val_check_interval"] = len(datamodule.train_dataloader()) // n_vals
    group_name = f"{multimodal_strategy}_{config['model']['protein_model']}_{config['model']['language_model']}"
    group_name = group_name.replace("/", "_")

    callbacks = []
    save_checkpoints = config["trainer"].pop("save_checkpoints", [])
    time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
    if "loss" in save_checkpoints:
        callbacks.append(
            ModelCheckpoint(
                monitor="loss/val_loss",
                mode="min",
                save_top_k=1,
                dirpath=f"{ROOT}/model_checkpoints/{time_stamp}_{group_name}_{config.get('seed', 'random')}_loss",
                verbose=True,
            )
        )
    if "f1" in save_checkpoints:
        callbacks.append(
            ModelCheckpoint(
                monitor="biochem/val_localization_f1",
                mode="max",
                save_top_k=1,
                dirpath=f"{ROOT}/model_checkpoints/{time_stamp}_{group_name}_{config.get('seed', 'random')}_f1",
                verbose=True,
            )
        )

    if log_to_wandb:
        callbacks.append(ExceptionHandlerCallback())
        config["wandb"]["group"] = group_name
        wandb_logger = WandbLogger(**config["wandb"])
        trainer = Trainer(logger=wandb_logger, callbacks=callbacks, **config["trainer"])
        trainer.logger.log_hyperparams(config)
        trainer.fit(model, datamodule)
        wandb.finish()
    else:
        trainer = Trainer(callbacks=callbacks, **config["trainer"])
        trainer.fit(model, datamodule)


if __name__ == "__main__":
    config = cli_parser()
    train_cprt(config, log_to_wandb=config.pop("log_to_wandb", False))
