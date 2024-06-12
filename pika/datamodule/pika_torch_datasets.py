import random
from typing import Dict, Literal, Tuple

import pandas as pd
from loguru import logger
from torch.utils.data import Dataset

from pika.utils.model_utils import get_is_real_question, shuffle_protein


class PikaDataset(Dataset[Tuple[str, str]]):
    """Pika torch dataset for training."""

    def __init__(self, ann_df: pd.DataFrame, sequences: Dict[str, str], split: Literal["train", "val", "test"]) -> None:
        """
        Initialize dataset.

        :param ann_df: must at least contain three columns:
                            - uniprot_id
                            - annotation: A set of strings defining the protein
                            - split: name of the split the id belongs
        :param sequences: dict of uniprot_ids mapped to the sequence
        :param split: split name
        """
        logger.info(f"preparing {split} dataset")
        self.split = split
        self.split_df = ann_df.loc[ann_df.split == self.split, ["uniprot_id", "annotation"]]
        self.sequences = {k: sequences[k] for k in self.split_df["uniprot_id"]}

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.split_df)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """Get Pika data for training."""
        uid, info = self.split_df.iloc[idx]
        protein_sequence = self.sequences[uid]
        if info == "control_question":
            # control question: whether the protein is real
            # randomly set the answer of half to No and shuffle the protein sequence
            if random.random() > 0.5:
                info = get_is_real_question(response="Yes")
            else:
                protein_sequence = shuffle_protein(protein_sequence)
                info = get_is_real_question(response="No")
        return protein_sequence, info


class PikaLiteDataset(Dataset[Tuple[str, str, str, str | int | bool]]):
    """Pika torch dataset for Biochem-Lite metrics."""

    def __init__(
        self,
        metrics_df: pd.DataFrame,
        sequences: Dict[str, str],
        split: Literal["train", "val", "test"],
    ) -> None:
        """
        Initialize dataset.

        :param metrics_df: must contain following columns:
                            - uniprot_id
                            - metric
                            - value
                            - split: name of the split the uniprot_id belongs
        :param sequences: dict of uniprot_ids mapped to the sequence
        :param split: split name
        """
        logger.info(f"preparing {split} metrics dataset")
        self.question_mapping = {
            "is_real": "Is this the sequence of a real protein?",
            "is_fake": "Is this a fake protein?",
            "is_enzyme": "Can this protein be considered an enzyme?",
            "in_membrane": "Does this protein localize to membranes?",
            "in_nucleus": "Does this protein localize to the nucleus?",
            "in_mitochondria": "Does this protein localize to mitochondria?",
            "localization": "What is the sub-cellular location of this protein?",
            "cofactor": "What is a cofactor of this protein?",
            "mw": "What is the molecular weight of this protein?",
        }
        self.split = split
        self.split_df = metrics_df[metrics_df.split == self.split].drop("split", axis=1)
        self.split_df = self.split_df[self.split_df["metric"].isin(self.question_mapping)]
        self.split_df = self.split_df.reset_index(drop=True)
        self.split_df = self.split_df[["uniprot_id", "metric", "value"]]

        self.sequences = {k: sequences[k] for k in self.split_df["uniprot_id"]}

        # add shuffled sequences for unreal proteins
        unreal_ids = self.split_df[(self.split_df["metric"] == "is_real") & (self.split_df["value"] == False)]
        # fixing shuffles to make outcomes comparable across steps
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


class PikaReActDataset(Dataset[Tuple[str, str, str, str]]):
    """Pika torch dataset for inference with Biochem-ReAct questions."""

    def __init__(self, react_df: pd.DataFrame, sequences: Dict[str, str], subject: str) -> None:
        """
        Initialize dataset.

        :param react_df: must at least contain three columns:
                            - uniprot_id
                            - subjects: name of the subject
                            - ground_truth: true answer to the subject of test
        :param sequences: dict of uniprot_ids mapped to the sequence
        :param subject: test subject
        """
        logger.info(f"preparing test dataset for {subject}")
        self.subject = subject
        question_mapping = {
            "catalytic activity": "What chemical reaction is catalyzed by this protein?",
            "cofactor": "What are the cofactors of this protein?",
            "functional domains": "What are the functional domains of this protein?",
        }
        self.question = question_mapping[subject]
        self.split_df = react_df[react_df.subjects == subject][["uniprot_id", "ground_truth"]]
        self.sequences = {k: sequences[k] for k in self.split_df["uniprot_id"]}

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.split_df)

    def __getitem__(self, idx: int) -> Tuple[str, str, str, str]:
        """Get Pika data for inference."""
        uid, answer = self.split_df.iloc[idx]
        protein_sequence = self.sequences[uid]
        return protein_sequence, self.question, f"{uid}: {self.subject}", answer
