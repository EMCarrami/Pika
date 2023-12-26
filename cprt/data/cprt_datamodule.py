from collections import namedtuple
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Tokenizer

from cprt.data.cprt_torch_datasets import CprtDataset, CprtMetricDataset
from cprt.data.data_utils import load_data_from_path, random_split_df

CprtData = namedtuple("CprtData", ["info", "info_mask", "protein", "labels"])
CprtMetricData = namedtuple("CprtMetricData", ["info", "protein", "metric_name", "expected_value"])


class CprtDataModule(LightningDataModule):  # type: ignore[misc]
    """Data module and collator for CprtData."""

    def __init__(
        self,
        data_dict_path: str,
        split_ratios: Tuple[float, float, float],
        protein_model: str,
        language_model: str,
        max_protein_length: int,
        max_text_length: int = 250,
        data_field_names: str | List[str] = "qa",
        use_unreal_proteins: bool = False,
        sequence_placeholder: str = "<protein sequence placeholder> ",
        subsample_data: int | float = 1.0,
        train_batch_size: int = 4,
        eval_batch_size: int | None = None,
        num_workers: int = 4,
    ) -> None:
        """
        Load tokenizers and instantiate datasets.

        :param data_dict_path: path to dict of uniprot_ids mapped to all info fields and the sequence
        :param split_ratios: ratio of train, val and test sets
        :param protein_model: esm model to use for tokenizer
        :param language_model: language model to use for tokenizer
        :param max_protein_length: max length of protein allowed
        :param max_text_length: max length of text allowed
        :param data_field_names: name of data fields to use for training (must be present in data_dict)
        :param use_unreal_proteins: whether to use shuffled proteins for training and metrics
        :param sequence_placeholder: string that is put ahead of all text to accumulate sequence embeddings.
                                will be ignored in loss computation by setting label to -100
        :param subsample_data: ratio of the data or number of samples to process.
        :param train_batch_size: train batch size
        :param eval_batch_size: size of val/test batch size. If unspecified will be 4 * train_batch_size
        :param num_workers: number of dataloader workers
        """
        super(CprtDataModule, self).__init__()
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.eval_batch_size = 4 * train_batch_size if eval_batch_size is None else eval_batch_size

        self.protein_tokenizer = AutoTokenizer.from_pretrained(f"facebook/{protein_model}")
        self.protein_tokenizer.model_max_length = max_protein_length
        self.text_tokenizer = GPT2Tokenizer.from_pretrained(language_model)
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        self.max_text_length = max_text_length
        self.placeholder_length = len(self.text_tokenizer(sequence_placeholder)["input_ids"])
        self.sequence_placeholder = sequence_placeholder

        if isinstance(data_field_names, str):
            data_field_names = [data_field_names]
        data_dict = load_data_from_path(data_dict_path)
        metadata = pd.DataFrame(data_dict.keys(), columns=["uniprot_id"])

        if isinstance(subsample_data, float) or subsample_data == 1:
            metadata = metadata.sample(frac=subsample_data)
        elif isinstance(subsample_data, int):
            metadata = metadata.sample(n=subsample_data)
        else:
            raise ValueError(f"subsample_data must be int or float. {subsample_data} was given.")

        metadata.loc[:, "protein_length"] = metadata["uniprot_id"].apply(lambda x: len(data_dict[x]["sequence"]))
        metadata = metadata[metadata["protein_length"] < max_protein_length]
        metadata.reset_index(drop=True, inplace=True)
        random_split_df(metadata, split_ratios)

        # ending with 'yes_real' means the question is about real proteins.
        # the questions should be ignored when use_unreal_proteins == False
        data_fields: Dict[str, List[str]] = {
            uid: [v for fn in data_field_names for v in fields[fn] if use_unreal_proteins or not v.endswith("yes_real")]
            for uid, fields in data_dict.items()
        }
        metadata.loc[:, "examples"] = metadata["uniprot_id"].apply(lambda x: data_fields[x])
        sequences: Dict[str, str] = {uid: v["sequence"] for uid, v in data_dict.items()}
        self.train_dataset = CprtDataset(metadata, sequences, "train")
        self.val_dataset = CprtDataset(metadata, sequences, "val")
        self.test_dataset = CprtDataset(metadata, sequences, "test")

        metrics_df = pd.DataFrame.from_dict({k: v["metrics"] for k, v in data_dict.items()}, orient="index")
        metrics_df.reset_index(inplace=True)
        metrics_df.rename(columns={"index": "uniprot_id"}, inplace=True)
        if use_unreal_proteins:
            metrics_df["is_real"] = np.random.choice([0, 1], size=len(metrics_df))
        metrics_df = pd.merge(metrics_df, metadata[["uniprot_id", "split"]], on="uniprot_id")
        self.val_metric_dataset = CprtMetricDataset(metrics_df, sequences, "val")

    def train_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Set up train loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.data_collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> Tuple[DataLoader, DataLoader]:  # type: ignore[type-arg]
        """Set up val loader."""
        return (
            DataLoader(
                self.val_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                collate_fn=self.data_collate_fn,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.val_metric_dataset,
                batch_size=self.eval_batch_size,
                shuffle=False,
                collate_fn=self.metric_collate_fn,
                num_workers=self.num_workers,
            ),
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Set up test loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collate_fn,
            num_workers=self.num_workers,
        )

    def data_collate_fn(self, batch: List[Tuple[str, str]]) -> CprtData:
        """Collate, pad and tokenize protein and info strings."""
        protein_sequences, info_list = zip(*batch)
        info_list = [f"{self.sequence_placeholder} {i}{self.text_tokenizer.eos_token}" for i in info_list]
        tokenized_info = self.text_tokenizer(
            info_list,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_text_length,
        )
        labels = tokenized_info["input_ids"].clone()
        labels[:, : self.placeholder_length] = -100
        for i, pad_idx in enumerate((1 - tokenized_info["attention_mask"]).sum(1)):
            if pad_idx > 0:
                labels[i, -pad_idx:] = -100
        return CprtData(
            info=tokenized_info["input_ids"],
            info_mask=tokenized_info["attention_mask"],
            protein=self.protein_tokenizer(protein_sequences, padding=True, return_tensors="pt")["input_ids"],
            labels=labels,
        )

    def metric_collate_fn(self, batch: List[Tuple[str, str, str, str | int]]) -> CprtMetricData:
        """Collate, pad and tokenize protein and metric information."""
        protein_sequences, questions, metric_list, values = zip(*batch)
        questions = [f"{self.sequence_placeholder} {i}" for i in questions]
        tokenized_questions = self.text_tokenizer(
            questions,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_text_length,
        )
        return CprtMetricData(
            info=tokenized_questions["input_ids"],
            protein=self.protein_tokenizer(protein_sequences, padding=True, return_tensors="pt")["input_ids"],
            metric_name=metric_list,
            expected_value=values,
        )
