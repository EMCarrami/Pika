from transformers import logging as transformers_logging

from pika.utils.helpers import cli_parser
from pika.utils.run_utils import train_classifier, train_pika

transformers_logging.set_verbosity_error()

if __name__ == "__main__":
    config = cli_parser()
    if "classifier" in config["model"]:
        train_classifier(config, log_to_wandb=config.pop("log_to_wandb", False))
    else:
        train_pika(config, log_to_wandb=config.pop("log_to_wandb", False))
