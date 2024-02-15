import csv
import os
from datetime import datetime
from typing import Any, Dict, List, Literal

import pandas as pd
import torch
from lightning import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from loguru import logger

import wandb
from pika.datamodule.pika_datamodule import PikaDataModule
from pika.model.pika_model import PikaModel
from pika.utils.checkpoint_utils import PartialCheckpointConnector, load_from_checkpoint
from pika.utils.helpers import ExceptionHandlerCallback


class Pika:
    """
    Class for Pika models' training and evaluation.

    Requires a config input as below:
    """

    def __init__(self, config: Dict[str, Any], inference_only: bool = False) -> None:
        """
        Initialize Pika module from config.

        :param inference_only: When True only model will be loaded/created without trainer or datamodule.
        """
        if "seed" in config:
            seed_everything(config["seed"])

        if "checkpoint" in config["model"]:
            self.model = load_from_checkpoint(config["model"]["checkpoint"])
            config["model"].update(dict(self.model.hparams))
            self.model_state = "pretrained"
        else:
            self.model = PikaModel(**config["model"])
            self.model_state = "init"

        if inference_only:
            logger.info("inference only module. Can only be used with the 'enquire' method")
        else:
            log_to_wandb = "wandb" in config
            multimodal_strategy = config["model"]["multimodal_strategy"]
            language_model = config["model"]["language_model"]
            protein_model = config["model"]["protein_model"]

            config["datamodule"]["language_model"] = language_model
            config["datamodule"]["protein_model"] = protein_model
            self.datamodule = PikaDataModule(**config["datamodule"])

            callbacks: List[Callback] = []
            if "checkpoint_callback" not in config:
                config["checkpoint_callback"] = {}
            time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
            checkpoint_path = config["checkpoint_callback"].get("checkpoint_path", "model_checkpoints")
            partial_checkpoint = config["checkpoint_callback"].get("save_partial_checkpoints", True)
            checkpoint_monitors = config["checkpoint_callback"].get("checkpoint_monitors", [])
            checkpoint_modes = config["checkpoint_callback"].get("checkpoint_modes", [])
            assert len(checkpoint_modes) == len(checkpoint_monitors), (
                "checkpoint_modes and checkpoint_monitors must be of the same size. "
                f"{checkpoint_modes} and {checkpoint_monitors} were given."
            )

            model_name = f"{multimodal_strategy}_{protein_model}_{language_model}".replace("/", "_")
            for monitor, mode in zip(checkpoint_monitors, checkpoint_modes):
                ckpt_name = monitor.split("/")[-1].split("_", 1)[-1]
                callbacks.append(
                    ModelCheckpoint(
                        monitor=monitor,
                        mode=mode,
                        save_top_k=1,
                        dirpath=f"{checkpoint_path}/{time_stamp}_{model_name}_{config.get('seed', 'rand')}_{ckpt_name}",
                        verbose=True,
                    )
                )

            if log_to_wandb:
                callbacks.append(ExceptionHandlerCallback())
                if config["wandb"].get("log_model", False):
                    assert len(checkpoint_monitors) > 0, "log_model==True for wandb without any checkpoint callbacks."
                self.trainer = Trainer(logger=WandbLogger(**config["wandb"]), callbacks=callbacks, **config["trainer"])
                self.trainer.logger.log_hyperparams(config)  # type: ignore[union-attr]
            else:
                self.trainer = Trainer(callbacks=callbacks, **config["trainer"])

            if partial_checkpoint and len(checkpoint_monitors) > 0:
                logger.info("will save partial model checkpoints. Ensure this is intended.")
                self.trainer._checkpoint_connector = PartialCheckpointConnector(self.trainer)

        self.config = config

    def train(self) -> None:
        """Train the Pika model."""
        self.trainer.fit(self.model, self.datamodule)
        self.model_state = "trained"

    @torch.no_grad()
    def biochem_react_benchmark(
        self,
        model_to_use: Literal["best", "last"] | None = None,
        file_save_path: str | None = None,
        wandb_config: Dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """
        Get Biochem-ReAct benchmark answers.

        Uses best/last checkpoint if called after training.
        When loaded from a checkpoint, uses the checkpoint model for inference.

        Returns a dataframe of responses (can also be written in file or logged to wandb.)
        """
        if file_save_path is not None:
            assert not os.path.isfile(file_save_path), f"{file_save_path} already exists. Use a new file name."
            assert (
                len(file_save_path.split(".")) == 2
            ), "use a valid file name with one file extension and no additional '.'s in the name."
            os.makedirs(os.path.dirname(file_save_path), exist_ok=True)

        assert self.datamodule.test_df is not None, (
            "datamodule was created without test subjects."
            "Replace obj.datamodule with a new instance of PikaDataModule with required test_subjects "
            "before calling obj.biochem_react_benchmark()"
        )

        self._update_model_state(model_to_use)
        self.trainer.test(self.model, self.datamodule)

        col_names = ["uniprot_id", "subject", "expected_answer", "generated_response"]
        if file_save_path is not None:
            with open(file_save_path, "a", newline="") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerow(col_names)
                for row in self.model.test_results:
                    writer.writerow(row)

        if wandb_config is not None:
            wandb.init(**wandb_config)
            test_table = wandb.Table(columns=col_names)
            for v in self.model.test_results:
                test_table.add_data(*v)
            wandb.log({"Biochem-ReAct_results": test_table})
            wandb.finish()

        return pd.DataFrame(self.model.test_results, columns=col_names)

    @torch.no_grad()
    def enquire(
        self,
        proteins: List[str] | str,
        question: str,
        generation_length: int = 30,
        keep_prompt: bool = False,
        protein_sequence_placeholder: str | None = None,
        model_to_use: Literal["best", "last"] | None = None,
    ) -> List[str]:
        """Generate answer to the question for a given protein sequence."""
        if isinstance(proteins, str):
            proteins = [proteins]
        self._update_model_state(model_to_use)
        if protein_sequence_placeholder is None:
            if "datamodule" in self.config and "sequence_placeholder" in self.config["datamodule"]:
                protein_sequence_placeholder = self.config["datamodule"]["sequence_placeholder"]
                logger.info(f"using datamodule's placeholder {protein_sequence_placeholder}")
            else:
                protein_sequence_placeholder = "<protein sequence placeholder> "
                logger.info(f"using default placeholder {protein_sequence_placeholder}. ")
        question = f"{protein_sequence_placeholder}{question}"
        return self.model.get_response(
            protein_ids=self.model.protein_tokenizer(proteins, padding=True, return_tensors="pt")["input_ids"],
            info_ids=self.model.text_tokenizer([question] * len(proteins), return_tensors="pt")["input_ids"],
            generation_length=generation_length,
            keep_prompt=keep_prompt,
        )

    def _update_model_state(self, model_to_use: Literal["best", "last"] | None) -> None:
        """Load best model if available and needed."""
        if self.model_state == "trained" and model_to_use == "best":
            assert hasattr(self.trainer.checkpoint_callback, "best_model_path"), (
                "best model not found in trained model. "
                "Specify checkpoint_modes and checkpoint_monitors for trainer or set model_to_use==last"
            )
            self.model = PikaModel.load_from_checkpoint(
                self.trainer.checkpoint_callback.best_model_path, strict=False  # type: ignore[union-attr]
            )
