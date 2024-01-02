import random
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class CPrtDataset(Dataset[Tuple[str, str]]):
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
                            - examples: A set of strings defining the protein
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
        if info.endswith("yes_real"):
            # question is about whether the protein is real
            # randomly set the answer of half to No and shuffle the protein sequence
            if random.random() > 0.5:
                info = info.replace("yes_real", "Yes")
            else:
                protein_sequence = shuffle_protein(protein_sequence)
                info = info.replace("yes_real", "No")
        return protein_sequence, info


class CPrtMetricDataset(Dataset[Tuple[str, str, str, str | int | bool]]):
    """Torch dataset class for biochem metrics."""

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
        self.question_mapping = {
            "is_real": "Is this the sequence of a real protein?",
            "is_fake": "Is this a fake protein?",
            "is_enzyme": "Is this protein an enzyme?",
            "is_enzyme_hard": "Can this protein be considered an enzyme?",
            "in_membrane": "Does this protein localize to membranes?",
            "in_nucleus": "Does this protein localize to the nucleus?",
            "in_mitochondria": "Does this protein localize to mitochondria?",
            "localization": "What is the sub-cellular location of this protein?",
            "kingdom": "To which kingdom of life does this protein belong?",
            "cofactor": "What is a cofactor of this protein?",
            "mw": "What is the molecular weight of this protein?",
        }
        # "length": "What is the length of this protein?",
        # "DNA_binding": "Does this protein bind to DNA?",
        # "RNA_binding": "Does this protein bind to RNA?",
        # "nucleic_acid_binding": "Does this protein bind nucleic acids?",
        self.split = split
        self.split_df = metadata[metadata.split == self.split].drop("split", axis=1)
        self.split_df["is_real"] = np.random.choice([False, True], size=len(self.split_df))
        self.split_df["is_fake"] = self.split_df["is_real"].map(lambda x: not x)
        self.split_df["is_enzyme_hard"] = self.split_df["is_enzyme"]
        self.sequences = {k: v for k, v in sequences.items() if k in self.split_df["uniprot_id"].to_list()}

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
        """Get Cprt metric data for training."""
        uid, metric, value = self.split_df.iloc[idx]
        if (metric == "is_real" and value is False) or (metric == "is_fake" and value is True):
            protein_sequence = self.sequences[f"{uid}_unreal"]
        else:
            protein_sequence = self.sequences[uid]
        question = self.question_mapping[metric]
        return protein_sequence, question, metric, value


def shuffle_protein(seq: str) -> str:
    """Shuffle the protein sequence except for the first and last 5 aa."""
    mid = list(seq[5:-5])
    random.shuffle(mid)
    return seq[:5] + "".join(mid) + seq[-5:]
