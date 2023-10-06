import json
from typing import Any, Dict

from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import wandb
from cprt.data.cprt_datamodule import CprtDataModule
from cprt.data.data_utils import load_data_from_path, random_split_df
from cprt.model.cprt_model import Cprt
from cprt.utils import ROOT


def train_cprt(config: Dict[str, Any]) -> None:
    """Run Cprt training."""
    if "seed" in config:
        seed_everything(config["seed"])

    data_dict = load_data_from_path(config["data"]["data_dict_path"])
    data_df = load_data_from_path(config["data"]["data_df_path"])
    data_df = data_df[
        data_df["protein_length"] < config["datamodule"]["max_protein_length"]
    ]
    random_split_df(data_df, config["data"]["split_ratios"])

    datamodule = CprtDataModule(data_dict, data_df, **config["datamodule"])
    model = Cprt(**config["model"])
    checkpoint_callback = ModelCheckpoint(
        monitor="loss/val_loss",
        mode="min",
        save_top_k=1,
        dirpath=f"{ROOT}/model_checkpoints/{config['model']['protein_model']}_{config['model']['language_model']}",
        verbose=True,
    )

    wandb.init(**config["wandb"])
    trainer = Trainer(
        logger=WandbLogger(), callbacks=[checkpoint_callback], **config["trainer"]
    )
    trainer.fit(model, datamodule)
    wandb.finish()


if __name__ == "__main__":
    with open(f"{ROOT}/configs/train_config.json", "r") as f:
        config: Dict[str, Any] = json.load(f)
    train_cprt(config)
