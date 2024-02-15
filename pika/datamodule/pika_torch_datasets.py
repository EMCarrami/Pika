import random
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset

from pika.datamodule.datamodule_helpers import shuffle_protein


class PikaDataset(Dataset[Tuple[str, str]]):
    """Pika torch dataset for training."""

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
                            - examples: A set of strings defining the protein
                            - split: name of the split the uniprot_id belongs
        :param sequences: dict of uniprot_ids mapped to the sequence
        :param split: split name
        """
        logger.info(f"preparing {split} dataset")
        self.split = split
        self.split_df = metadata[metadata.split == self.split][["uniprot_id", "examples"]]
        self.sequences = {k: sequences[k] for k in self.split_df["uniprot_id"]}
        self.split_df = self.split_df.explode("examples", ignore_index=True)

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.split_df)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """Get Pika data for training."""
        uid, info = self.split_df.iloc[idx]
        protein_sequence = self.sequences[uid]
        if info.endswith("yes_real"):
            # question is about whether the protein is real
            # randomly set the answer of half to No and shuffle the protein sequence
            if random.random() > 0.5:
                info = info.replace("yes_real", "Yes")
            else:
                protein_sequence = shuffle_protein(protein_sequence)
                info = info.replace("yes_real", "No")
        return protein_sequence, info


class PikaReActDataset(Dataset[Tuple[str, str, str, str]]):
    """Pika torch dataset for inference with Biochem-ReAct questions."""

    def __init__(self, metadata: pd.DataFrame, sequences: Dict[str, str], subject: str) -> None:
        """
        Initialize dataset.

        :param metadata: must at least contain three columns:
                            - uniprot_id
                            - subjects: name of the subject
                            - ground_truth: true answer to the subject of test
                            - split: name of the split the uniprot_id belongs
        :param sequences: dict of uniprot_ids mapped to the sequence
        :param subject: test subject
        """
        logger.info(f"preparing test dataset for {subject}")
        self.subject = subject
        question_mapping = {
            "catalytic activity": "What chemical reaction is catalyzed by this protein?",
            "cofactor": "What are the cofactors of this protein?",
            "functional domains": "What are the functional domains of this protein?",
            "taxonomy": "To what taxonomic group or organism does this protein belong?",
        }
        self.question = question_mapping[subject]
        self.split_df = metadata[metadata.subjects == subject][["uniprot_id", "ground_truth"]]
        self.sequences = {k: sequences[k] for k in self.split_df["uniprot_id"]}

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.split_df)

    def __getitem__(self, idx: int) -> Tuple[str, str, str, str]:
        """Get Pika data for inference."""
        uid, answer = self.split_df.iloc[idx]
        protein_sequence = self.sequences[uid]
        return protein_sequence, self.question, f"{uid}: {self.subject}", answer


class PikaLiteDataset(Dataset[Tuple[str, str, str, str | int | bool]]):
    """Pika torch dataset for Biochem-Lite metrics."""

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
                            - examples: A set of strings defining the protein
                            - split: name of the split the uniprot_id belongs
        :param sequences: dict of uniprot_ids mapped to the sequence
        :param split: split name
        """
        logger.info(f"preparing {split} metrics dataset")
        self.question_mapping = {
            "is_real": "Is this the sequence of a real protein?",
            "is_fake": "Is this a fake protein?",
            "is_enzyme": "Is this protein an enzyme?",
            "is_enzyme_hard": "Can this protein be considered an enzyme?",
            "in_membrane": "Does this protein localize to membranes?",
            "in_nucleus": "Does this protein localize to the nucleus?",
            "in_mitochondria": "Does this protein localize to mitochondria?",
            "localization": "What is the sub-cellular location of this protein?",
            "cofactor": "What is a cofactor of this protein?",
            "mw": "What is the molecular weight of this protein?",
        }
        self.split = split
        self.split_df = metadata[metadata.split == self.split].drop("split", axis=1)
        self.split_df["is_real"] = np.random.choice([False, True], size=len(self.split_df))
        self.split_df["is_fake"] = self.split_df["is_real"].map(lambda x: not x)
        self.split_df["is_enzyme_hard"] = self.split_df["is_enzyme"]
        self.sequences = {k: sequences[k] for k in self.split_df["uniprot_id"]}

        self.split_df = self.split_df.melt(id_vars=["uniprot_id"], var_name="metric", value_name="value")
        # Filter questions
        self.split_df = self.split_df[self.split_df["metric"].isin(self.question_mapping)]
        self.split_df = self.split_df[self.split_df["value"] != "none"]
        self.split_df = self.split_df[self.split_df["value"] != "Viruses"]
        self.split_df = self.split_df.reset_index(drop=True)

        # add shuffled sequences for unreal proteins
        unreal_ids = self.split_df[(self.split_df["metric"] == "is_real") & (self.split_df["value"] == False)]
        self.sequences |= {f"{uid}_unreal": shuffle_protein(self.sequences[uid]) for uid in unreal_ids["uniprot_id"]}

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.split_df)

    def __getitem__(self, idx: int) -> Tuple[str, str, str, str | int | bool]:
        """Get Pika-Lite data for metrics."""
        uid, metric, value = self.split_df.iloc[idx]
        if (metric == "is_real" and value == False) or (metric == "is_fake" and value == True):
            protein_sequence = self.sequences[f"{uid}_unreal"]
        else:
            protein_sequence = self.sequences[uid]
        question = self.question_mapping[metric]
        return protein_sequence, question, metric, value
