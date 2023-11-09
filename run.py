import json
from typing import Any, Dict

import wandb
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from transformers import logging as transformers_logging

from cprt.data.datamodule_factory import creat_datamodule
from cprt.model.cprt_model import Cprt
from cprt.utils import ROOT

transformers_logging.set_verbosity_error()


def train_cprt(config: Dict[str, Any], log_to_wandb: bool = False) -> None:
    """Run Cprt training."""
    if "seed" in config:
        seed_everything(config["seed"])

    datamodule = creat_datamodule(**config["data"], datamodule_config=config["datamodule"], only_keep_questions=True)
    model = Cprt(**config["model"])
    checkpoint_callback = ModelCheckpoint(
        monitor="loss/val_loss",
        mode="min",
        save_top_k=1,
        dirpath=f"{ROOT}/model_checkpoints/{config['model']['protein_model']}_{config['model']['language_model']}",
        verbose=True,
    )

    config["trainer"]["log_every_n_steps"] = 1
    if log_to_wandb:
        wandb.init(**config["wandb"])
        trainer = Trainer(logger=WandbLogger(), callbacks=[checkpoint_callback], **config["trainer"])
        trainer.logger.log_hyperparams(config)
        try:
            trainer.fit(model, datamodule)
        except Exception as e:
            print(f"An error occurred: {e}")
            model.log_wandb_table()
        finally:
            model.log_wandb_table()
        wandb.finish()
    else:
        trainer = Trainer(callbacks=[checkpoint_callback], **config["trainer"])
        trainer.fit(model, datamodule)


if __name__ == "__main__":
    with open(f"{ROOT}/configs/train_config.json", "r") as f:
        config: Dict[str, Any] = json.load(f)
    train_cprt(config, log_to_wandb=False)
