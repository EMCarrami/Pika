from datetime import datetime
from typing import Any, Dict

import wandb
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from pika.baselines.classification_datamodule import ClassificationDataModule
from pika.baselines.classification_model import ProteinClassificationModel
from pika.datamodule.pika_datamodule import PikaDataModule
from pika.model.pika_model import PikaModel
from pika.utils import ROOT
from pika.utils.helpers import ExceptionHandlerCallback


def train_pika(config: Dict[str, Any], log_to_wandb: bool = False) -> None:
    """Run Pika training."""
    if "seed" in config:
        seed_everything(config["seed"])

    config["datamodule"]["language_model"] = config["model"]["language_model"]
    config["datamodule"]["protein_model"] = config["model"]["protein_model"]

    if "[seed]" in config["datamodule"]["data_dict_path"]:
        config["datamodule"]["data_dict_path"] = config["datamodule"]["data_dict_path"].replace(
            "[seed]", str(config["seed"])
        )
    datamodule = PikaDataModule(**config["datamodule"])

    multimodal_strategy = config["model"]["multimodal_strategy"]
    model = PikaModel(**config["model"])

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
