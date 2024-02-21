from transformers import logging as transformers_logging

from pika.main import Pika
from pika.utils.helpers import cli_parser
from pika.utils.run_utils import train_classifier

transformers_logging.set_verbosity_error()

if __name__ == "__main__":
    config, run_mode = cli_parser()
    if run_mode == "train_classifier":
        train_classifier(config)
    elif run_mode == "train":
        model = Pika(config)
        model.train()
    elif run_mode == "train_and_benchmark":
        model = Pika(config)
        model.train()
        model.biochem_react_benchmark(model_to_use="best")
    elif run_mode == "benchmark_only":
        assert "checkpoint" in config["model"], "checkpoint path must be provided in model config for benchmark_only"
        model = Pika(config)
        model.biochem_react_benchmark()
    elif run_mode == "infer_only" or run_mode == "enquire":
        enq_config = config.pop("enquiry")
        assert (
            "proteins" in enq_config and "question" in enq_config
        ), "must specify enquiry.proteins and enquiry.question in config for inference_only/enquiry mode."
        model = Pika(config, inference_only=True)
        model.enquire(**enq_config)
    else:
        raise ValueError("--run_mode must be correctly specified.")
