import json
import os
import unittest
from time import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

from pika.utils.helpers import cli_parser


class TestSimpleParser(unittest.TestCase):
    """Test class for simple_parser."""

    mock_config_path: str
    mock_config: Dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        """Create a mock JSON config file."""
        cls.mock_config = {
            "seed": 123,
            "datamodule": {"sequence_placeholder": "<prt> "},
            "model": {"language_model": "gpt2", "protein_layer_to_use": -1},
        }
        cls.mock_config_path = f"{time()}_mock_config.json"
        with open(cls.mock_config_path, "w") as file:
            json.dump(cls.mock_config, file)

    @patch("argparse.ArgumentParser.parse_known_args")
    def test_simple_parser(self, mock_args: MagicMock) -> None:
        """Perform tests for valid and invalid input values."""
        base_args = ["--config", self.mock_config_path]

        # Fail without run_mode or wrong one
        mock_args.return_value = ([], [])
        with self.assertRaises(AssertionError):
            cli_parser()
        mock_args.return_value = ([], base_args)
        with self.assertRaises(AssertionError):
            cli_parser()
        mock_args.return_value = ([], base_args + ["--run_mode", "wrong"])
        with self.assertRaises(AssertionError):
            cli_parser()

        # Test arg updates
        base_args += ["--run_mode", "train"]
        mock_args.return_value = ([], base_args + ["--seed", "0"])
        expected_config_1 = self.mock_config.copy()
        expected_config_1["seed"] = 0
        self.assertEqual(cli_parser(), (expected_config_1, "train"))

        mock_args.return_value = (
            [],
            base_args + ["--datamodule.sequence_placeholder", " ", "--model.protein_layer_to_use", "all"],
        )
        expected_config_2 = self.mock_config.copy()
        expected_config_2["datamodule"]["sequence_placeholder"] = " "
        expected_config_2["model"]["protein_layer_to_use"] = "all"
        self.assertEqual(cli_parser(), (expected_config_2, "train"))

        # Test invalid arguments
        mock_args.return_value = ([], base_args + ["--invalid-format", "--value"])
        with self.assertRaises(AssertionError):
            cli_parser()
        mock_args.return_value = ([], base_args + ["no_key", "value"])
        with self.assertRaises(AssertionError):
            cli_parser()
        mock_args.return_value = ([], base_args + ["--missing.key", "value"])
        with self.assertRaises(KeyError):
            cli_parser()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up the mock config file after tests."""
        os.remove(cls.mock_config_path)
