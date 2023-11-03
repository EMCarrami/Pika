import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict

import torch.cuda
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from wandb.wandb_run import Run

import wandb
from cprt.data.cprt_datamodule import CprtDataModule
from cprt.data.datamodule_factory import creat_datamodule
from cprt.model.cprt_model import Cprt
from cprt.utils import ROOT
from cprt.utils.helpers import gpu_usage_logger


def fit_and_log(trainer: Trainer, model: Cprt, datamodule: CprtDataModule) -> None:
    """Ensure table is logged on Exception."""
    try:
        trainer.fit(model, datamodule)
    except Exception as e:
        print(f"An error occurred: {e}")
        model.log_table()
    finally:
        model.log_table()


def train_cprt(config: Dict[str, Any], log_to_wandb: bool = False, monitor_gpu: bool = False) -> None:
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
        if monitor_gpu and torch.cuda.is_available():
            assert isinstance(wandb.run, Run)
            wandb_config = {
                "project": wandb.run.project,
                "id": wandb.run.id,
                "name": wandb.run.name,
            }
            stop_event = multiprocessing.Event()
            gpu_id = int(trainer.strategy.root_device.split(":")[1])
            with ProcessPoolExecutor(max_workers=2) as executor:
                training = executor.submit(fit_and_log, trainer, model, datamodule)
                gpu_usage = executor.submit(gpu_usage_logger, wandb_config, stop_event, gpu_id, log_interval=1)
                for future in as_completed([training, gpu_usage]):
                    if future == training:
                        stop_event.set()
        else:
            fit_and_log(trainer, model, datamodule)
        wandb.finish()
    else:
        trainer = Trainer(callbacks=[checkpoint_callback], **config["trainer"])
        trainer.fit(model, datamodule)


if __name__ == "__main__":
    with open(f"{ROOT}/configs/train_config.json", "r") as f:
        config: Dict[str, Any] = json.load(f)
    train_cprt(config, log_to_wandb=True, monitor_gpu=True)
