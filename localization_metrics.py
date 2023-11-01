import argparse
import csv
import json
from collections import defaultdict
from typing import Any, Dict, Tuple

import numpy as np
import torch
from lightning.pytorch import seed_everything

from cprt.data.cprt_datamodule import CprtDataModule
from cprt.data.datamodule_factory import creat_datamodule
from cprt.model.cprt_model import Cprt
from cprt.utils import ROOT

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_from_checkpoint(config_path: str, checkpoint_path: str) -> Tuple[Cprt, CprtDataModule]:
    """Load model and datamodule from checkpoint and config."""
    with open(config_path, "r") as f:
        config: Dict[str, Any] = json.load(f)
    if "seed" in config:
        seed_everything(config["seed"])
    datamodule = creat_datamodule(**config["data"], datamodule_config=config["datamodule"], only_keep_questions=True)
    model = Cprt.load_from_checkpoint(checkpoint_path, map_location=device)
    return model, datamodule


def localization_metrics(
    config_path: str,
    checkpoint_path: str,
    locations: Tuple[str, ...] = ("membrane", "nucleus", "mitochondria"),
    question: str = "What is the subcellular location of this protein?",
    generation_length: int = 50,
    output_csv: str | None = None,
) -> None:
    """Compute localization prediction accuracy on test set."""
    if output_csv is None:
        output_csv = checkpoint_path.split("/")[-1].replace(".ckpt", ".csv")

    model, datamodule = load_from_checkpoint(config_path, checkpoint_path)
    model.eval()
    test_loader = datamodule.test_dataloader()

    seqs, loc_info = defaultdict(list), defaultdict(list)
    for batch in test_loader:
        for info, seq in zip(batch.info, batch.protein):
            info_text = datamodule.text_tokenizer.decode(info, skip_special_tokens=True)
            if "where" in info_text.split("?")[0].lower() or "location" in info_text.split("?")[0]:
                for lc in locations:
                    if lc in info_text:
                        seqs[lc].append(seq)
                        loc_info[lc].append(info_text)

    tokenized_question = datamodule.text_tokenizer(
        [f"{datamodule.sequence_placeholder} {question}"], return_tensors="pt"
    )

    accuracy = defaultdict(list)
    results = []
    for lc in locations:
        print(f"predicting {lc}, {len(seqs[lc])}")
        for seq, expected in zip(seqs[lc], loc_info[lc]):
            with torch.no_grad():
                preds = model.cprt_llm.generate(
                    tokenized_question["input_ids"].to(device),
                    encoder_hidden_states=model.esm(seq.unsqueeze(0).to(device)),
                    use_cache=False,
                    max_length=generation_length,
                )
            response = datamodule.text_tokenizer.decode(preds[0].cpu())
            results.append((lc, expected, response))
            if lc in response:
                accuracy[lc].append(1)
            else:
                accuracy[lc].append(0)
        print(f"{lc} accuracy: {np.mean(accuracy[lc])}")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default=f"{ROOT}/configs/train_config.json", help="path to config used in training"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What is the subcellular location of this protein?",
        help="Question prompt to use.",
    )
    parser.add_argument("--checkpoint_path", type=int, help="path to model checkpoint")
    args = parser.parse_args()

    localization_metrics(config_path=args.config_path, checkpoint_path=args.checkpoint_path, question=args.question)
