from collections import namedtuple
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from cprt.data.cprt_torch_datasets import shuffle_protein
from cprt.data.data_utils import load_data_from_path, random_split_df

ClassificationData = namedtuple("ClassificationData", ["protein_ids", "labels"])


class ClassificationDataModule(LightningDataModule):  # type: ignore[misc]
    """Data module and collator for CPrtData."""

    def __init__(
        self,
        data_dict_path: str,
        split_ratios: Tuple[float, float, float],
        classification_task: str,
        protein_model: str,
        max_protein_length: int,
        min_protein_length: int = 0,
        train_batch_size: int = 4,
        eval_batch_size: int | None = None,
        num_workers: int = 4,
    ) -> None:
        """
        Load tokenizers and instantiate datasets.

        :param data_dict_path: path to dict of uniprot_ids mapped to all info fields and the sequence
        :param split_ratios: ratio of train, val and test sets
        :param classification_task: can be one of is_enzyme, is_real, localization, kingdom, mw
        :param protein_model: esm model to use for tokenizer
        :param max_protein_length: max length of protein allowed
        :param min_protein_length: min protein length to use. Useful for debugging GPU OOM
        :param train_batch_size: train batch size
        :param eval_batch_size: size of val/test batch size. If unspecified will be 4 * train_batch_size
        :param num_workers: number of dataloader workers
        """
        super(ClassificationDataModule, self).__init__()
        assert classification_task in [
            "is_enzyme",
            "is_real",
            "localization",
            "kingdom",
            "mw",
        ], "classification_task should be one of is_enzyme, is_real, localization, kingdom, mw."
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.eval_batch_size = 4 * train_batch_size if eval_batch_size is None else eval_batch_size

        self.protein_tokenizer = AutoTokenizer.from_pretrained(f"facebook/{protein_model}")
        self.protein_tokenizer.model_max_length = max_protein_length

        data_dict = load_data_from_path(data_dict_path)
        metadata = pd.DataFrame(data_dict.keys(), columns=["uniprot_id"])

        # TDOD: remove later
        metadata = metadata.sample(frac=1.0)

        sequences: Dict[str, str] = {uid: v["sequence"] for uid, v in data_dict.items()}

        metadata.loc[:, "protein_length"] = metadata["uniprot_id"].apply(lambda x: len(data_dict[x]["sequence"]))
        metadata.loc[:, "uniref_id"] = metadata["uniprot_id"].apply(lambda x: data_dict[x]["uniref_id"])
        metadata = metadata[metadata["protein_length"] < max_protein_length]
        metadata = metadata[metadata["protein_length"] > min_protein_length]
        metadata.reset_index(drop=True, inplace=True)
        random_split_df(metadata, split_ratios, key="uniref_id")

        metrics_df = pd.DataFrame.from_dict({k: v["metrics"] for k, v in data_dict.items()}, orient="index")
        metrics_df.reset_index(inplace=True)
        metrics_df.rename(columns={"index": "uniprot_id"}, inplace=True)
        metrics_df = pd.merge(metrics_df, metadata[["uniprot_id", "split"]], on="uniprot_id")
        metrics_df["is_real"] = np.random.choice([False, True], size=len(metrics_df))

        metrics_df = metrics_df[metrics_df[classification_task] != "none"]
        metrics_df = metrics_df[metrics_df[classification_task] != "Viruses"]
        metrics_df = metrics_df.reset_index(drop=True)

        if classification_task == "mw":
            self.num_classes = 1
            metrics_df["class_id"] = np.log10(metrics_df[classification_task]).astype(np.float32)
        else:
            metrics_df["class_id"], uniques = pd.factorize(metrics_df[classification_task])
            self.num_classes = len(uniques)

        metrics_df = metrics_df[["uniprot_id", "class_id", classification_task, "split"]]

        self.train_dataset = ClassificationDataset(metrics_df, classification_task, sequences, "train")
        self.val_dataset = ClassificationDataset(metrics_df, classification_task, sequences, "val")
        self.test_dataset = ClassificationDataset(metrics_df, classification_task, sequences, "test")

    def train_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Set up train loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Set up val loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Set up test loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=self.num_workers,
        )

    def collator(self, batch: List[Tuple[str, str]]) -> ClassificationData:
        """Collate, pad and tokenize protein sequences."""
        protein_sequences, labels = zip(*batch)
        return ClassificationData(
            protein_ids=self.protein_tokenizer(protein_sequences, padding=True, return_tensors="pt")["input_ids"],
            labels=torch.tensor(labels),
        )


class ClassificationDataset(Dataset[Tuple[str, int]]):
    """Torch dataset class for biochem metrics."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        classification_task: str,
        sequences: Dict[str, str],
        split: Literal["train", "val", "test"],
    ) -> None:
        """
        Initialize dataset.

        :param metadata: must at least contain three columns:
                            - uniprot_id
                            - [classification_task]: name of the task
                            - class_id: class the protein belongs
                            - split: name of the split the uniprot_id belongs
        :param sequences: dict of uniprot_ids mapped to the sequence
        :param split: split name
        """
        self.split = split
        self.split_df = metadata[metadata.split == self.split].drop("split", axis=1)
        self.sequences = {k: sequences[k] for k in self.split_df["uniprot_id"]}

        self.make_unreal = False
        if classification_task == "is_real":
            self.make_unreal = True
            unreal_ids = self.split_df[self.split_df["is_real"] == False]
            self.sequences |= {
                f"{uid}_unreal": shuffle_protein(self.sequences[uid]) for uid in unreal_ids["uniprot_id"]
            }
        self.split_df = self.split_df[["uniprot_id", "class_id", classification_task]]

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.split_df)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        """Get classification data for training and eval."""
        uid, class_id, label = self.split_df.iloc[idx]
        if self.make_unreal and label is False:
            protein_sequence = self.sequences[f"{uid}_unreal"]
        else:
            protein_sequence = self.sequences[uid]
        return protein_sequence, class_id
