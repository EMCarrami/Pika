import os
import unittest
from typing import Dict

import numpy as np
import pandas as pd

from pika.datamodule.pika_torch_datasets import (
    PikaDataset,
    PikaLiteDataset,
    PikaReActDataset,
)


class TestPikaDatasets(unittest.TestCase):
    """Test Pika Datasets for getitem and protein shuffling."""

    metadata: pd.DataFrame
    sequences: Dict[str, str]

    @classmethod
    def setUpClass(cls) -> None:
        super(TestPikaDatasets, cls).setUpClass()
        assets_path = os.path.join(os.path.dirname(__file__), "../assets")
        sample_sequences_path = os.path.join(assets_path, "sample_sequences.csv")
        cls.metadata = pd.read_csv(sample_sequences_path)
        cls.sequences = cls.metadata.set_index("uniprot_id")["sequence"].to_dict()

    def test_pika_dataset(self) -> None:
        data = self.metadata.copy()
        data["split"] = "train"
        data["annotation"] = "control_question"
        dataset = PikaDataset(data, self.sequences, "train")
        # test sizes and types
        item = dataset[0]
        self.assertIsInstance(item, tuple)
        self.assertIsInstance(item[0], str)
        self.assertIsInstance(item[1], str)
        self.assertEqual(len(item), 2)
        # test protein shuffle
        answers = []
        for item in iter(dataset):
            answer = item[1].split()[-1]
            self.assertIn(answer, ["Yes", "No"], f"answer not Yes/No: {answer}")
            answers.append(answer)
            if answer == "Yes":
                self.assertIn(item[0], self.sequences.values())
            elif answer == "No":
                # shuffled protein must be absent
                self.assertNotIn(item[0], self.sequences.values())
            else:
                raise ValueError(f"answer for {item} is invalid.")
        self.assertIn("Yes", answers)
        self.assertIn("No", answers)

    def test_pika_lite_dataset(self) -> None:
        data = self.metadata.copy()
        data["split"] = "val"
        data["metric"] = "is_real"
        data["value"] = np.random.choice([False, True], size=len(data))
        data2 = data.copy()
        data2["metric"] = "is_fake"
        data2["value"] = data["value"].map(lambda x: not x)
        dataset = PikaLiteDataset(pd.concat([data, data2], ignore_index=True), self.sequences, "val")
        # test sizes and types
        item = dataset[0]
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 4)
        self.assertIsInstance(item[0], str)
        self.assertIsInstance(item[1], str)
        self.assertIsInstance(item[2], str)
        self.assertIsInstance(item[3], np.bool_)
        # test protein shuffle
        answers = []
        for item in iter(dataset):
            answer = item[3]
            self.assertIn(answer, [True, False], f"answer not True/False: {answer}")
            answers.append(answer)
            if (item[2] == "is_real" and answer == True) or (item[2] == "is_fake" and answer == False):
                self.assertIn(item[0], self.sequences.values())
            elif (item[2] == "is_real" and answer == False) or (item[2] == "is_fake" and answer == True):
                # shuffled protein must be absent in original sequences, but present in dataset's
                self.assertNotIn(item[0], self.sequences.values())
                self.assertIn(item[0], dataset.sequences.values())
            else:
                raise ValueError(f"{item} is invalid.")

    def test_pika_react_dataset(self) -> None:
        data = self.metadata.copy()
        data["ground_truth"] = "mock, cofactor"
        data["subjects"] = "cofactor"
        dataset = PikaReActDataset(data, self.sequences, "cofactor")
        # test sizes and types
        item = dataset[0]
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 4)
        for _item in item:
            self.assertIsInstance(_item, str)
        for item in iter(dataset):
            self.assertIn(item[2].split(":")[0], self.metadata["uniprot_id"].to_list())
