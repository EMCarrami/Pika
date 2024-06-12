from collections import namedtuple
from typing import Dict, List, Tuple

import numpy as np
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
        sequence_data_path: str,
        annotations_path: str,
        metrics_data_path: str,
        split_path: str,
        protein_model: str,
        language_model: str,
        max_protein_length: int,
        min_protein_length: int = 30,
        max_text_length: int = 250,
        data_types_to_use: str | List[str] = "qa",
        add_control_question: bool = True,
        sequence_placeholder: str = "<protein sequence placeholder> ",
        train_batch_size: int = 4,
        eval_batch_size: int | None = None,
        test_subjects: str | List[str] | None = None,
        num_workers: int = 4,
    ) -> None:
        """
        Load tokenizers and instantiate datasets.

        :param sequence_data_path: path to sequences csv file with "uniprot_id" and "sequence" columns
        :param annotations_path: path to annotations csv file with "uniprot_id" and "annotation" columns
        :param metrics_data_path: path to metrics csv file with "uniprot_id", "metric" and "value" columns
        :param split_path: path to csv with mapping of uniprot_id to split: [train, val, test]
        :param protein_model: esm model to use for tokenizer
        :param language_model: language model to use for tokenizer
        :param max_protein_length: max length of protein allowed
        :param min_protein_length: min protein length to use.
        :param max_text_length: max length of text allowed
        :param data_types_to_use: name of data fields to use for training (must be present in data_dict)
        :param add_control_question: whether to add control question as an additional example
        :param sequence_placeholder: string that is put ahead of all text to accumulate sequence embeddings.
                                will be ignored in loss computation by setting label to -100
        :param train_batch_size: train batch size
        :param eval_batch_size: size of val/test batch size. If unspecified will be 4 * train_batch_size
        :param test_subjects: scientific subjects for which tests must be performed.
                        Supports: "reaction", "cofactor", "domains" or "all"
                        Creates a test dataloader for each subject.
        :param num_workers: number of dataloader workers
        """
        super(PikaDataModule, self).__init__()
        rev = "main"
        if "phi" in language_model:
            rev = "7e10f3ea09c0ebd373aebc73bc6e6ca58204628d"
        if isinstance(data_types_to_use, str):
            data_types_to_use = [data_types_to_use]

        # load data
        logger.info(f"loading data from {sequence_data_path}, {annotations_path} & {metrics_data_path}")
        metadata, ann_df, metrics_df = self._load_data(
            sequence_data_path, annotations_path, metrics_data_path, split_path
        )

        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.eval_batch_size = 4 * train_batch_size if eval_batch_size is None else eval_batch_size
        self.test_subjects = test_subjects

        self.protein_tokenizer = AutoTokenizer.from_pretrained(f"facebook/{protein_model}")
        self.protein_tokenizer.model_max_length = max_protein_length + 2
        self.text_tokenizer = AutoTokenizer.from_pretrained(language_model, revision=rev)
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        self.max_text_length = max_text_length

        # duplicate text_tokenizer is to allow for parallel tokenization using fast tokenizers
        _text_tokenizer = AutoTokenizer.from_pretrained(language_model, use_fast=False, revision=rev)
        self.placeholder_length = len(_text_tokenizer(sequence_placeholder)["input_ids"])
        self.sequence_placeholder = sequence_placeholder

        # apply protein length filters
        if "length" not in metadata.columns:
            metadata.loc[:, "length"] = metadata["sequence"].apply(len)
        metadata = metadata[metadata["length"] <= max_protein_length]
        metadata = metadata[metadata["length"] >= min_protein_length]
        ann_df = ann_df[ann_df["uniprot_id"].isin(metadata["uniprot_id"].to_list())]
        metrics_df = metrics_df[metrics_df["uniprot_id"].isin(metadata["uniprot_id"].to_list())]
        metadata.reset_index(drop=True, inplace=True)
        metrics_df.reset_index(drop=True, inplace=True)
        sequences: Dict[str, str] = metadata.set_index("uniprot_id")["sequence"].to_dict()

        # prepare test sets if needed
        self.test_df = None
        if self.test_subjects is not None:
            allowed = ["catalytic activity", "reaction", "cofactor", "domains", "functional domains"]
            if isinstance(self.test_subjects, str):
                self.test_subjects = allowed if self.test_subjects == "all" else [self.test_subjects]
            assert all([i in allowed for i in self.test_subjects]), f"only {allowed} test_subjects are supported."
            map_subjects = {"reaction": "catalytic activity", "domains": "functional domains"}
            self.test_subjects = list(set([map_subjects.get(i, i) for i in self.test_subjects]))
            self.test_df = ann_df.loc[
                (ann_df.type == "fields") & (ann_df.split == "test"), ["uniprot_id", "annotation"]
            ]
            # keep only test subjects
            self.test_df = self.test_df[
                self.test_df["annotation"].apply(lambda x: any([x.startswith(s) for s in self.test_subjects]))
            ]
            self.test_df = self.test_df.reset_index(drop=True, inplace=False)
            self.test_df[["subjects", "ground_truth"]] = self.test_df["annotation"].str.split(":", n=1, expand=True)
            self.test_df = self.test_df[["uniprot_id", "subjects", "ground_truth"]]
            self.test_sequences = {k: sequences[k] for k in self.test_df["uniprot_id"]}

        # merge all data fields for train and val sets
        logger.info("preparing examples")
        ann_df = ann_df[ann_df.type.isin(data_types_to_use)]
        ann_df.reset_index(drop=True, inplace=True)
        if add_control_question:
            _cq_df = ann_df.drop_duplicates(subset="uniprot_id", keep="first").copy()
            _cq_df["type"] = "control"
            _cq_df["annotation"] = "control_question"
            ann_df = pd.concat([ann_df, _cq_df], ignore_index=True)
        self.train_dataset = PikaDataset(ann_df, sequences, "train")
        self.val_dataset = PikaDataset(ann_df, sequences, "val")

        if add_control_question:
            _cq_real = metrics_df.drop_duplicates(subset="uniprot_id", keep="first").copy()
            _cq_fake = metrics_df.drop_duplicates(subset="uniprot_id", keep="first").copy()
            _cq_real["metric"] = "is_real"
            _cq_fake["metric"] = "is_fake"
            _cq_real["value"] = np.random.choice([False, True], size=len(_cq_real))
            _cq_fake["value"] = _cq_real["value"].apply(lambda x: not x)
            metrics_df = pd.concat([metrics_df, _cq_real, _cq_fake], ignore_index=True)
        self.val_metric_dataset = PikaLiteDataset(metrics_df, sequences, "val")

    @staticmethod
    def _load_data(
        sequence_data_path: str, annotations_path: str, metrics_data_path: str, split_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load data and check input data are correct."""
        metadata = pd.read_csv(sequence_data_path)
        ann_df = pd.read_csv(annotations_path)
        metrics_df = pd.read_csv(metrics_data_path)
        try:
            splits_map = pd.read_csv(split_path).set_index("uniprot_id")["split"].to_dict()
        except KeyError:
            raise AssertionError(
                f"{split_path} must be a csv file with column headers providing [uniprot_id, split]. "
                f"Some are missing in {pd.read_csv(split_path).columns}"
            )
        # add split to metadata
        metadata["split"] = metadata["uniprot_id"].map(splits_map)
        ann_df["split"] = ann_df["uniprot_id"].map(splits_map)
        metrics_df["split"] = metrics_df["uniprot_id"].map(splits_map)
        # check data are correct
        all_data = (metadata, ann_df, metrics_df)
        paths = (sequence_data_path, annotations_path, metrics_data_path)
        expected_columns = (["uniprot_id", "sequence"], ["uniprot_id", "annotation"], ["uniprot_id", "metric", "value"])
        assert (
            ann_df["split"].notna().all()
        ), f"{split_path} must provide split values for all ids in {annotations_path}"
        for d, e, p in zip(all_data, expected_columns, paths):
            assert all(
                [_e in d.columns for _e in e]
            ), f"{p} must be a csv file with column headers providing {e}. Some are missing in {d.columns}"
        seq_ids = set(metadata["uniprot_id"].to_list())
        ann_ids = set(ann_df["uniprot_id"].to_list())
        metric_ids = set(metrics_df["uniprot_id"].to_list())
        assert len(ann_ids - seq_ids) == 0, f"all ids in {annotations_path} must be present in {sequence_data_path}"
        assert len(metric_ids - ann_ids) == 0, f"all ids in {metrics_data_path} must be present in {annotations_path}"
        return all_data

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
