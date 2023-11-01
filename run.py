import json
from typing import Any, Dict

from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from cprt.data.datamodule_factory import creat_datamodule
from cprt.model.cprt_model import Cprt
from cprt.utils import ROOT


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

    if log_to_wandb:
        wandb.init(**config["wandb"])
        trainer = Trainer(logger=WandbLogger(), callbacks=[checkpoint_callback], **config["trainer"])
        trainer.logger.log_hyperparams(config)
        trainer.fit(model, datamodule)
        wandb.finish()
    else:
        trainer = Trainer(callbacks=[checkpoint_callback], **config["trainer"])
        trainer.fit(model, datamodule)


if __name__ == "__main__":
    with open(f"{ROOT}/configs/train_config.json", "r") as f:
        config: Dict[str, Any] = json.load(f)
    train_cprt(config, log_to_wandb=True)
