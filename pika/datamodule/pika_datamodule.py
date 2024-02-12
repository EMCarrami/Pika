import pickle
from collections import namedtuple
from typing import Any, Dict, List, Tuple

import pandas as pd
from lightning import LightningDataModule
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from pika.datamodule.pika_torch_datasets import (
    PikaDataset,
    PikaLiteDataset,
    PikaReActDataset,
)

PikaData = namedtuple("PikaData", ["protein", "info", "info_mask", "labels"])
PikaMetricData = namedtuple("PikaMetricData", ["protein", "question", "metric_name", "expected_value"])


class PikaDataModule(LightningDataModule):
    """Data module and collator for PikaData."""

    test_df: pd.DataFrame | None

    def __init__(
        self,
        data_dict_path: str,
        split_path: str,
        protein_model: str,
        language_model: str,
        max_protein_length: int,
        min_protein_length: int = 0,
        max_text_length: int = 250,
        data_field_names: str | List[str] = "qa",
        sequence_placeholder: str = "<protein sequence placeholder> ",
        train_batch_size: int = 4,
        eval_batch_size: int | None = None,
        test_subjects: str | List[str] | None = None,
        num_workers: int = 4,
    ) -> None:
        """
        Load tokenizers and instantiate datasets.

        :param data_dict_path: path to dict of uniprot_ids mapped to all info fields and the sequence
        :param split_path: path to csv with mapping of uniprot_id to split: [train, val, test]
        :param protein_model: esm model to use for tokenizer
        :param language_model: language model to use for tokenizer
        :param max_protein_length: max length of protein allowed
        :param min_protein_length: min protein length to use. Useful for debugging GPU OOM
        :param max_text_length: max length of text allowed
        :param data_field_names: name of data fields to use for training (must be present in data_dict)
        :param sequence_placeholder: string that is put ahead of all text to accumulate sequence embeddings.
                                will be ignored in loss computation by setting label to -100
        :param train_batch_size: train batch size
        :param eval_batch_size: size of val/test batch size. If unspecified will be 4 * train_batch_size
        :param test_subjects: scientific subjects for which tests must be performed.
                        Supports: "reaction", "cofactor", "domains", "taxonomy" and "all"
                        Creates a test dataloader for each subject.
        :param num_workers: number of dataloader workers
        """
        super(PikaDataModule, self).__init__()
        rev = "main"
        if "phi" in language_model:
            rev = "7e10f3ea09c0ebd373aebc73bc6e6ca58204628d"
        if isinstance(data_field_names, str):
            data_field_names = [data_field_names]

        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.eval_batch_size = 4 * train_batch_size if eval_batch_size is None else eval_batch_size
        self.test_subjects = test_subjects

        self.protein_tokenizer = AutoTokenizer.from_pretrained(f"facebook/{protein_model}")
        self.protein_tokenizer.model_max_length = max_protein_length
        self.text_tokenizer = AutoTokenizer.from_pretrained(language_model, revision=rev)
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        self.max_text_length = max_text_length

        # duplicate text_tokenizer is to allow for parallel tokenization using fast tokenizers
        _text_tokenizer = AutoTokenizer.from_pretrained(language_model, use_fast=False, revision=rev)
        self.placeholder_length = len(_text_tokenizer(sequence_placeholder)["input_ids"])
        self.sequence_placeholder = sequence_placeholder

        # load data
        logger.info(f"loading data from {data_dict_path}")
        with open(data_dict_path, "rb") as f:
            data_dict: Dict[str, Any] = pickle.load(f)
        sequences: Dict[str, str] = {uid: v["sequence"] for uid, v in data_dict.items()}

        # set up splits
        logger.info(f"loading splits from {split_path}")
        metadata = pd.read_csv(split_path)
        assert (
            "uniprot_id" in metadata.columns and "split" in metadata.columns
        ), "split_path must be a csv file with column headers providing uniprot_id and split"
        assert all([i in data_dict for i in metadata["uniprot_id"]]), (
            f"all uniprot_id of {split_path} must be present in {data_dict_path}. "
            f"missing keys: {set(metadata['uniprot_id'].to_list()) - set(data_dict.keys())}"
        )
        metadata.loc[:, "protein_length"] = metadata["uniprot_id"].apply(lambda x: len(data_dict[x]["sequence"]))
        metadata.loc[:, "uniref_id"] = metadata["uniprot_id"].apply(lambda x: data_dict[x]["uniref_id"])
        metadata = metadata[metadata["protein_length"] < max_protein_length]
        metadata = metadata[metadata["protein_length"] > min_protein_length]
        metadata.reset_index(drop=True, inplace=True)

        # prepare test sets if needed
        self.test_df = None
        if self.test_subjects is not None:
            allowed = ["catalytic activity", "reaction", "cofactor", "domains", "functional domains", "taxonomy"]
            self.test_subjects = allowed if self.test_subjects == "all" else self.test_subjects
            assert all([i in allowed for i in self.test_subjects]), f"only {allowed} test_subjects are supported."
            map_subjects = {"reaction": "catalytic activity", "domains": "functional domains"}
            self.test_subjects = list(set([map_subjects.get(i, i) for i in self.test_subjects]))
            self.test_df = metadata[metadata.split == "test"].copy()
            self.test_df["subjects"] = self.test_df["uniprot_id"].map(
                lambda x: [i for i in data_dict[x]["fields"] if any([i.startswith(s) for s in self.test_subjects])]
            )
            self.test_df = self.test_df[self.test_df["subjects"].apply(lambda x: len(x) > 0)]
            self.test_df = self.test_df.explode("subjects", ignore_index=True)
            self.test_df[["subjects", "ground_truth"]] = self.test_df["subjects"].str.split(":", n=1, expand=True)
            self.test_sequences = {k: sequences[k] for k in self.test_df["uniprot_id"]}

        # merge all data fields for train and val sets
        logger.info("preparing examples")
        data_fields: Dict[str, List[str]] = {
            uid: [v for fn in data_field_names for v in fields[fn]] for uid, fields in data_dict.items()
        }
        metadata.loc[:, "examples"] = metadata["uniprot_id"].apply(lambda x: data_fields[x])
        self.train_dataset = PikaDataset(metadata, sequences, "train")
        self.val_dataset = PikaDataset(metadata, sequences, "val")

        metrics_df = pd.DataFrame.from_dict({k: v["metrics"] for k, v in data_dict.items()}, orient="index")
        metrics_df.reset_index(inplace=True)
        metrics_df.rename(columns={"index": "uniprot_id"}, inplace=True)

        metrics_df = pd.merge(metrics_df, metadata[["uniprot_id", "split"]], on="uniprot_id")
        self.val_metric_dataset = PikaLiteDataset(metrics_df, sequences, "val")

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

    def test_dataloader(self) -> Tuple[DataLoader, ...]:  # type: ignore[type-arg]
        """Set up a test dataloader for each test_subject."""
        assert (
            self.test_subjects is not None and self.test_df is not None and len(self.test_df) > 0
        ), "test_subjects must be provided for test_dataloader"
        return tuple(
            [
                DataLoader(
                    PikaReActDataset(self.test_df, self.test_sequences, subject),
                    batch_size=self.eval_batch_size,
                    collate_fn=self.metric_collate_fn,
                    num_workers=self.num_workers,
                )
                for subject in self.test_subjects
            ]
        )

    def data_collate_fn(self, batch: List[Tuple[str, str]]) -> PikaData:
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
        return PikaData(
            protein=self.protein_tokenizer(protein_sequences, padding=True, return_tensors="pt")["input_ids"],
            info=tokenized_info["input_ids"],
            info_mask=tokenized_info["attention_mask"],
            labels=labels,
        )

    def metric_collate_fn(self, batch: List[Tuple[str, str, str, str | int]]) -> PikaMetricData:
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
        return PikaMetricData(
            protein=self.protein_tokenizer(protein_sequences, padding=True, return_tensors="pt")["input_ids"],
            question=tokenized_questions["input_ids"],
            metric_name=metric_list,
            expected_value=values,
        )
