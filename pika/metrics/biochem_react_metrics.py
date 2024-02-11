import csv
from typing import Any, Dict

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from pika.utils.checkpoint_utils import load_from_checkpoint
from pika.utils.helpers import get_output_file_path


def run_biochem_react_metrics(config: Dict[str, Any]) -> None:
    """Run test for Biochem-ReAct with pretrained model and log to wandb or file."""
    checkpoint = config["checkpoint"]["path"]
    model, data = load_from_checkpoint(config, checkpoint, is_partial=config["checkpoint"].get("is_partial", False))
    out_file = get_output_file_path(config)
    if "wandb" in config:
        wandb_logger = WandbLogger(**config["wandb"])
        trainer = Trainer(logger=wandb_logger, **config["trainer"])
        assert isinstance(trainer.logger, WandbLogger)
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
