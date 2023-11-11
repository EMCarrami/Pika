import json
from typing import Any, Dict

import wandb
from lightning import Callback, Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from transformers import logging as transformers_logging

from cprt.data.datamodule_factory import creat_datamodule
from cprt.model import CPRT_MODELS
from cprt.model.cprt_model import BaseCPrtModel
from cprt.utils import ROOT

transformers_logging.set_verbosity_error()


class ExceptionHandlerCallback(Callback):  # type: ignore[misc]
    """Callback to handle exceptions."""

    def on_exception(self, trainer: Trainer, model: BaseCPrtModel, exception: Exception) -> None:
        """Run final logs."""
        model.on_train_end()


def train_cprt(config: Dict[str, Any], log_to_wandb: bool = False) -> None:
    """Run Cprt training."""
    if "seed" in config:
        seed_everything(config["seed"])

    datamodule = creat_datamodule(**config["data"], datamodule_config=config["datamodule"], only_keep_questions=True)

    multimodal_strategy = config["model"].pop("multimodal_strategy")
    model = CPRT_MODELS[multimodal_strategy](**config["model"])

    group_name = f"{multimodal_strategy}_{config['model']['protein_model']}_{config['model']['language_model']}"
    checkpoint_callback = ModelCheckpoint(
        monitor="loss/val_loss",
        mode="min",
        save_top_k=1,
        dirpath=f"{ROOT}/model_checkpoints/{group_name}_{config.get('seed', 'random')}",
        verbose=True,
    )

    if log_to_wandb:
        config["wandb"]["group"] = group_name
        wandb_logger = WandbLogger(**config["wandb"])
        trainer = Trainer(
            logger=wandb_logger, callbacks=[checkpoint_callback, ExceptionHandlerCallback()], **config["trainer"]
        )
        trainer.logger.log_hyperparams(config)
        trainer.fit(model, datamodule)
        wandb.finish()
    else:
        trainer = Trainer(callbacks=checkpoint_callback, **config["trainer"])
        trainer.fit(model, datamodule)


if __name__ == "__main__":
    with open(f"{ROOT}/configs/train_config.json", "r") as f:
        config: Dict[str, Any] = json.load(f)
    train_cprt(config, log_to_wandb=False)
