from glob import glob
from typing import Any, Dict

import torch
import wandb
from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)
from loguru import logger

from pika.model.pika_model import PikaModel

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class PartialCheckpointConnector(_CheckpointConnector):
    """Custom CheckpointConnector for saving models partially."""

    def dump_checkpoint(self, weights_only: bool = False) -> Dict[str, Any]:
        """Override dump_checkpoint to only save trained weights."""
        # Generate the checkpoint using the parent class method
        checkpoint = super().dump_checkpoint(weights_only)

        # only keep weights with gradient
        assert isinstance(self.trainer.model, PikaModel)
        trainable_params = [name for name, param in self.trainer.model.named_parameters() if param.requires_grad]
        checkpoint["state_dict"] = {k: v for k, v in checkpoint["state_dict"].items() if k in trainable_params}
        checkpoint["is_partial"] = True
        return checkpoint


def load_from_checkpoint(checkpoint_path: str) -> PikaModel:
    """Load model from checkpoint."""
    if checkpoint_path.startswith("wandb:"):
        checkpoint_path = find_wandb_checkpoint(checkpoint_path.replace("wandb:", ""))
    is_partial: bool = torch.load(checkpoint_path, map_location=torch.device("cpu")).get("is_partial", False)
    model: PikaModel = PikaModel.load_from_checkpoint(checkpoint_path, strict=not is_partial)
    logger.info(f"{'partial' if is_partial else ''} model loaded from checkpoint at {checkpoint_path}")
    return model


def find_wandb_checkpoint(checkpoint: str) -> str:
    """Download the model artifact into model_checkpoints and return checkpoint path."""
    api = wandb.Api()
    artifact = api.artifact(checkpoint, type="model")
    model_dir = checkpoint.split("/")[-1].replace(":", "_")
    artifact_dir: str = artifact.download(f"model_checkpoints/{model_dir}")
    ckpt_file = glob(f"{artifact_dir}/*.ckpt")
    assert len(ckpt_file) == 1, f"there are more than 1 ckpt file in {artifact_dir}: {ckpt_file}"
    logger.info(f"using wandb model checkpoint in {ckpt_file[0]}")
    return ckpt_file[0]
