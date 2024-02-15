import os
import unittest
from typing import Any, List, NamedTuple, Type

from torch import Tensor

from pika.datamodule.pika_datamodule import PikaData, PikaDataModule, PikaMetricData
from pika.utils.model_utils import CONTROL_QUESTIONS


class TestPikaDataModule(unittest.TestCase):
    """Test class for Pika Datamodule."""

    sample_data_path: str
    sample_split_path: str
    batch_size: int
    datamodule: PikaDataModule

    @classmethod
    def setUpClass(cls) -> None:
        super(TestPikaDataModule, cls).setUpClass()
        assets_path = os.path.join(os.path.dirname(__file__), "assets")
        cls.sample_data_path = os.path.join(assets_path, "sample_data.pkl")
        cls.sample_split_path = os.path.join(assets_path, "sample_split.csv")
        cls.batch_size = 2
        cls.datamodule = PikaDataModule(
            data_dict_path=cls.sample_data_path,
            split_path=cls.sample_split_path,
            language_model="gpt2",
            protein_model="esm2_t6_8M_UR50D",
            max_protein_length=1500,
            train_batch_size=cls.batch_size,
            eval_batch_size=cls.batch_size,
            test_subjects=["domains", "reaction", "taxonomy"],
        )

    def test_train_dataloader(self) -> None:
        train_loader = self.datamodule.train_dataloader()
        batch = next(iter(train_loader))
        # check batch is of current type and has the right attributes and size
        self.assertIsInstance(batch, PikaData)
        self._check_batch(batch, ["protein", "info", "info_mask", "labels"], [Tensor, Tensor, Tensor, Tensor])
        # test labels have eos token
        self.assertEqual(
            batch.labels.eq(self.datamodule.text_tokenizer.eos_token_id).sum().item(),
            self.batch_size,
            "labels dont seem to have exactly one eos token.",
        )

    def test_val_dataloader(self) -> None:
        val_loaders = self.datamodule.val_dataloader()
        # check test loader size matches subjects
        self.assertIsInstance(val_loaders, tuple)
        self.assertEqual(len(val_loaders), 2)
        # check first val loader: for linguistic metrics
        batch_0 = next(iter(val_loaders[0]))
        self.assertIsInstance(batch_0, PikaData)
        self._check_batch(batch_0, ["protein", "info", "info_mask", "labels"], [Tensor, Tensor, Tensor, Tensor])
        # check second val loader for NaNs:
        val1_df = self.datamodule.val_dataloader()[1].dataset.split_df  # type: ignore[attr-defined]
        self.assertNotIn("nan", [str(i) for i in val1_df.value.to_list()])
        # check second val loader: for Biochem-Lite metrics
        batch_1 = next(iter(val_loaders[1]))
        self.assertIsInstance(batch_1, PikaMetricData)
        self._check_batch(
            batch_1,
            ["protein", "question", "metric_name", "expected_value"],
            [Tensor, Tensor, str, [str, int, bool]],
        )

    def test_test_dataloader(self) -> None:
        test_loaders = self.datamodule.test_dataloader()
        # check test loader size matches subjects
        self.assertIsInstance(test_loaders, tuple)
        self.assertEqual(len(test_loaders), 3)
        batch = next(iter(test_loaders[0]))
        # check batch is of current type and has the right attributes and size
        self.assertIsInstance(batch, PikaMetricData)
        self._check_batch(batch, ["protein", "question", "metric_name", "expected_value"], [Tensor, Tensor, str, str])
        # check metric_names are correct
        eg_name = batch.metric_name[0]
        self.assertEqual(
            len(eg_name.split(":")), 2, "metric_name must be composed of : separated uniprot_id and metric_name"
        )

    def _check_batch(
        self, batch: NamedTuple, attributes: List[str], element_types: List[Type[Any] | List[Type[Any]]]
    ) -> None:
        """Check batch_size and that all elements of a batch exists and match expected types."""
        for i, t in zip(attributes, element_types):
            self.assertTrue(hasattr(batch, i), f"batch does not have a {i} attribute.")
            self.assertEqual(len(getattr(batch, i)), self.batch_size, f"incorrect batch size {i} in {batch}")
            if isinstance(t, list):
                self.assertTrue(any([isinstance(getattr(batch, i)[0], _t) for _t in t]), f"type missmatch for {i}")
            else:
                self.assertIsInstance(getattr(batch, i)[0], t)

    def test_control_question_added(self) -> None:
        """Test if control_question is added correctly to metadata for datasets."""
        # check correct fields when control question needed
        train_df = self.datamodule.train_dataset.split_df
        val0_df = self.datamodule.val_dataset.split_df
        val1_df = self.datamodule.val_metric_dataset.split_df
        self.assertIn("control_question", train_df["examples"].to_list())
        self.assertIn("control_question", val0_df["examples"].to_list())
        self.assertIn("is_real", val1_df["metric"].to_list())
        # ensure fields aren't there when control question not needed
        neg_dm = PikaDataModule(
            data_dict_path=self.sample_data_path,
            split_path=self.sample_split_path,
            language_model="gpt2",
            protein_model="esm2_t6_8M_UR50D",
            max_protein_length=1500,
            train_batch_size=self.batch_size,
            eval_batch_size=self.batch_size,
            add_control_question=False,
        )
        train_df = neg_dm.train_dataset.split_df
        val0_df = neg_dm.val_dataset.split_df
        val1_df = neg_dm.val_metric_dataset.split_df
        self.assertNotIn("control_question", train_df["examples"].to_list())
        self.assertNotIn("control_question", val0_df["examples"].to_list())
        self.assertNotIn("is_real", val1_df["metric"].to_list())

    def test_control_question_encoded(self) -> None:
        """Test if control_question is correctly created and used."""
        # check correct fields when control question needed
        original_df = self.datamodule.train_dataset.split_df
        # only keep control questions and reassign in datamodule
        original_df = original_df[original_df.examples == "control_question"]
        self.datamodule.train_dataset.split_df = original_df
        train_loader = self.datamodule.train_dataloader()
        batch = next(iter(train_loader))
        questions = self.datamodule.text_tokenizer.batch_decode(batch.info)
        for q in questions:
            self.assertTrue(any([i in q for i in CONTROL_QUESTIONS]), f"control question not found in {q}")
