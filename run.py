from transformers import logging as transformers_logging

from pika.main import Pika
from pika.utils.helpers import cli_parser
from pika.utils.run_utils import train_classifier

transformers_logging.set_verbosity_error()

if __name__ == "__main__":
    config = cli_parser()
    if "classifier" in config["model"]:
        train_classifier(config, log_to_wandb=config.pop("log_to_wandb", False))
    else:
        model = Pika(config)
        model.train()
