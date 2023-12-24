from typing import Dict, Literal, Tuple

import pandas as pd
from torch.utils.data import Dataset


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
        return protein_sequence, info


class CprtMetricDataset(Dataset[Tuple[str, str, str, str | int]]):
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
        self.split = split
        self.split_df = metadata[metadata.split == self.split].drop("split", axis=1)
        self.sequences = {k: v for k, v in sequences.items() if k in self.split_df["uniprot_id"].to_list()}

        self.split_df = self.split_df.melt(id_vars=["uniprot_id"], var_name="metric", value_name="value")
        self.split_df = self.split_df[~((self.split_df["metric"] == "cofactor") & (self.split_df["value"] == "none"))]
        self.split_df = self.split_df[
            ~((self.split_df["metric"] == "localization") & (self.split_df["value"] == "none"))
        ]
        self.split_df = self.split_df.reset_index(drop=True)

        self.question_mapping = {
            "DNA_binding": "Does this protein bind to DNA?",
            "RNA_binding": "Does this protein bind to RNA?",
            "nucleic_acid_binding": "Does this protein bind nucleic acids?",
            "is_enzyme": "Is this protein an enzyme?",
            "in_membrane": "Does this protein localize to membranes?",
            "in_nucleus": "Does this protein localize to the nucleus?",
            "in_mitochondria": "Does this protein localize to mitochondria?",
            "localization": "What is the sub-cellular location of this protein?",
            "kingdom": "To which kingdom of life does this protein belong?",
            "cofactor": "What is a cofactor of this protein?",
        }

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.split_df)

    def __getitem__(self, idx: int) -> Tuple[str, str, str, str | int]:
        """Get Cprt metric data for training."""
        uid, metric, value = self.split_df.iloc[idx]
        protein_sequence = self.sequences[uid]
        return protein_sequence, self.question_mapping[metric], metric, value
