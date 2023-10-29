from collections import namedtuple
from typing import Dict, List, Literal, Tuple

import pandas as pd
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, GPT2Tokenizer

CprtData = namedtuple("CprtData", ["protein", "info", "info_mask", "labels"])


class CprtDataset(Dataset[Tuple[str, str]]):
    """Cprt torch dataset."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        sequences: Dict[str, str],
        split: Literal["train", "val", "test"],
    ) -> None:
        """
        Initialize dataset.

        :param metadata: must at least contain three columns:
                            - uniprot_id
                            - examples: A Q/A string for the protein
                            - split: name of the split the uniprot_id belongs
        :param sequences: dict of uniprot_ids mapped to the sequence
        :param split: split name
        """
        self.split = split
        self.split_df = metadata[metadata.split == self.split][["uniprot_id", "examples"]]
        self.sequences = {k: v for k, v in sequences.items() if k in self.split_df["uniprot_id"].to_list()}
        self.split_df = self.split_df.explode("examples", ignore_index=True)

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.split_df)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """Get Cprt data for training."""
        uid, info = self.split_df.iloc[idx]
        protein_sequence = self.sequences[uid]
        return protein_sequence, info


class CprtDataModule(LightningDataModule):  # type: ignore[misc]
    """Data module and collator for CprtData."""

    def __init__(
        self,
        data_dict: Dict[str, Dict[str, List[str] | str]],
        metadata: pd.DataFrame,
        esm_model: str = "esm2_t6_8M_UR50D",
        language_model: str = "gpt2",
        batch_size: int = 4,
        max_protein_length: int = 1500,
        sequence_placeholder: str = "[protein sequence placeholder] ",
    ) -> None:
        """
        Load tokenziers and instantiate prepare datasets.

        :param data_dict: dict of uniprot_ids mapped to all info fields and the sequence
        :param metadata: must at least contain three columns:
                            - uniprot_id
                            - info_fields: List of info columns for the given uniprot_id
                            - split: name of the split the uniprot_id belongs
        :param esm_model: esm model to use for tokenizer
        :param language_model: language model to use for tokenizer
        :param batch_size: train, val and test main batch size
        :param max_protein_length: mex length of protein allowed for tokenizer to throw a warning
        :param sequence_placeholder: string that is put ahead of all text to accumulate sequence embeddings.
                                        will be ignored in loss computation by setting label to -100
        """
        super().__init__()
        self.batch_size = batch_size

        self.protein_tokenizer = AutoTokenizer.from_pretrained(f"facebook/{esm_model}")
        self.protein_tokenizer.model_max_length = max_protein_length
        self.text_tokenizer = GPT2Tokenizer.from_pretrained(language_model)
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        self.placeholder_length = len(self.text_tokenizer(sequence_placeholder)["input_ids"])
        self.sequence_placeholder = sequence_placeholder

        metadata.loc[:, "examples"] = metadata["uniprot_id"].apply(lambda x: data_dict[x]["info"])
        sequences: Dict[str, str] = {k: v["sequence"] for k, v in data_dict.items()}  # type: ignore[misc]

        self.train_dataset = CprtDataset(metadata, sequences, "train")
        self.val_dataset = CprtDataset(metadata, sequences, "val")
        self.test_dataset = CprtDataset(metadata, sequences, "test")

    def train_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Set up train loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=4,
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Set up val loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4,
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        """Set up test loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4,
        )

    def collate_fn(self, batch: List[Tuple[str, str]]) -> CprtData:
        """Collate, pad and tokenize protein and info strings."""
        protein_sequences, info_list = zip(*batch)
        info_list = [f"{self.sequence_placeholder} {i}{self.text_tokenizer.eos_token}" for i in info_list]
        tokenized_info = self.text_tokenizer(
            info_list,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        labels = tokenized_info["input_ids"][:, 1:].contiguous()
        labels[:, : self.placeholder_length] = -100
        for i, pad_idx in enumerate((1 - tokenized_info["attention_mask"]).sum(1)):
            if pad_idx > 0:
                labels[i, -pad_idx:] = -100
        return CprtData(
            info=tokenized_info["input_ids"][:, :-1].contiguous(),
            info_mask=tokenized_info["attention_mask"][:, :-1].contiguous(),
            protein=self.protein_tokenizer(protein_sequences, padding=True, return_tensors="pt")["input_ids"],
            labels=labels,
        )
