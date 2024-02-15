import os
import pickle
import unittest
from typing import Any, Dict

import numpy as np
import pandas as pd

from pika.datamodule.pika_torch_datasets import (
    PikaDataset,
    PikaLiteDataset,
    PikaReActDataset,
)


class TestPikaDatasets(unittest.TestCase):
    """Test Pika Datasets for getitem and protein shuffling."""

    data_dict: Dict[str, Dict[str, Any]]
    sequences: Dict[str, str]

    @classmethod
    def setUpClass(cls) -> None:
        super(TestPikaDatasets, cls).setUpClass()
        assets_path = os.path.join(os.path.dirname(__file__), "assets")
        sample_data_path = os.path.join(assets_path, "sample_data.pkl")

        with open(sample_data_path, "rb") as f:
            cls.data_dict = pickle.load(f)
        cls.sequences = {uid: v["sequence"] for uid, v in cls.data_dict.items()}

    def test_pika_dataset(self) -> None:
        metadata = pd.DataFrame(self.data_dict.keys(), columns=["uniprot_id"])
        metadata["split"] = "train"
        metadata["examples"] = [["control_question"] for _ in range(len(metadata))]
        dataset = PikaDataset(metadata, self.sequences, "train")
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
        metadata = pd.DataFrame(self.data_dict.keys(), columns=["uniprot_id"])
        metadata["split"] = "val"
        metadata["is_real"] = np.random.choice([False, True], size=len(metadata))
        metadata["is_fake"] = metadata["is_real"].map(lambda x: not x)
        dataset = PikaLiteDataset(metadata, self.sequences, "val")
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
        metadata = pd.DataFrame(self.data_dict.keys(), columns=["uniprot_id"])
        metadata["ground_truth"] = "mock, taxonomy"
        metadata["subjects"] = "taxonomy"
        dataset = PikaReActDataset(metadata, self.sequences, "taxonomy")
        # test sizes and types
        item = dataset[0]
        self.assertIsInstance(item, tuple)
        self.assertEqual(len(item), 4)
        for _item in item:
            self.assertIsInstance(_item, str)
        for item in iter(dataset):
            self.assertIn(item[2].split(":")[0], self.data_dict)
