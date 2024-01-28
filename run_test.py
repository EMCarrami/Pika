import csv
import json
import os.path
from typing import Any, Dict, Tuple

import torch
from lightning import Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from loguru import logger

import wandb
from cprt.data.cprt_datamodule import CPrtDataModule
from cprt.model.cprt_model import CPrtModel
from cprt.utils.helpers import cli_parser

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def run_test(config: Dict[str, Any]) -> None:
    """Run test with pretrained model and log to wandb."""
    checkpoint = config["checkpoint"]["path"]
    model, data = load_from_checkpoint(config, checkpoint, is_partial=config["checkpoint"].get("is_partial", False))
    out_file = get_output_file_path(config)
    if "wandb" in config:
        wandb_logger = WandbLogger(**config["wandb"])
        trainer = Trainer(logger=wandb_logger, **config["trainer"])
        trainer.logger.log_hyperparams(config)
    else:
        trainer = Trainer(**config["trainer"])
    trainer.test(model, data)
    if out_file:
        with open(out_file, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(("uniprot_id", "subject", "expected_answer", "generated_response"))
            for row in model.test_results:
                writer.writerow(row)


def get_output_file_path(config: Dict[str, Any]) -> str:
    """Get output file path and create directories."""
    if "save_file_path" in config:
        out_file = config["save_file_path"]
        if out_file == "auto":
            save_dir = "test_results"
            file_name = config["checkpoint"]["path"].split("/")[-1].split(".")[0]
            if "name" in config["wandb"]:
                file_name = f"{config['wandb']['name']}_{file_name.split('_')[-1]}.tsv"
            out_file = f"{save_dir}/{file_name}.tsv"
        else:
            assert out_file.endswith(".tsv"), "only csv format is supported"
            if len(out_file.split("/")) > 1:
                save_dir = "/".join(out_file.split("/")[:-1])
            else:
                save_dir = "test_results"
                out_file = f"{save_dir}/{out_file}"
        assert not os.path.isfile(out_file), f"{out_file} already exists"
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"predicted texts will be saved in {out_file}")
        assert isinstance(out_file, str)
        return out_file
    else:
        return ""


def load_from_checkpoint(
    config: Dict[str, Any], checkpoint_path: str, is_partial: bool = False
) -> Tuple[CPrtModel, CPrtDataModule]:
    """Load model and datamodule from checkpoint and config."""
    if os.path.isfile("model_checkpoints/model_checkpoints.json"):
        with open("model_checkpoints/model_checkpoints.json", "r") as file:
            checkpoints = json.load(file)

    if checkpoint_path in checkpoints["path"]:
        artifact_dir = find_wandb_checkpoint(checkpoints["path"][checkpoint_path])
        config["seed"] = checkpoints["seed"][checkpoint_path]
        logger.info(f"seed was automatically set to {config['seed']} for wandb model {checkpoint_path}")
        checkpoint_path = f"{artifact_dir}/{checkpoint_path}"

    if is_partial:
        model = load_reduced_model(checkpoint_path)
    else:
        model = CPrtModel.load_from_checkpoint(checkpoint_path, map_location=device)
    config["model"] = dict(model.hparams)
    config["datamodule"]["language_model"] = model.hparams["language_model"]
    config["datamodule"]["protein_model"] = model.hparams["protein_model"]
    if "seed" in config:
        seed_everything(config["seed"])
    datamodule = CPrtDataModule(**config["datamodule"])
    return model, datamodule


def find_wandb_checkpoint(checkpoint: str) -> str:
    """Download the model artifact into model_checkpoints."""
    api = wandb.Api()
    artifact = api.artifact(checkpoint, type="model")  # type: ignore[no-untyped-call]
    artifact_dir: str = artifact.download("model_checkpoints")
    return artifact_dir


def load_reduced_model(checkpoint_path: str) -> CPrtModel:
    """Load a reduced CPrtModel from checkpoint with weights for only trained parts."""
    nck = torch.load(checkpoint_path, map_location=device)
    model = CPrtModel(**nck["hyper_parameters"])
    missing = model.load_state_dict(nck["state_dict"], strict=False)
    assert len(missing.unexpected_keys) == 0
    return model


if __name__ == "__main__":
    config = cli_parser()
    run_test(config)
