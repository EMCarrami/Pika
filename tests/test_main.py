import os
import unittest

from pika.main import Pika
from pika.utils.helpers import load_config


class TestPika(unittest.TestCase):
    """Test class for Pika main model training and inference."""

    def test_self_pika(self) -> None:
        self._run_training("sample_self_pika_config.json")

    def test_cross_pika(self) -> None:
        self._run_training("sample_cross_pika_config.json")

    def _run_training(self, config_path: str) -> Pika:
        """Run training and return Pika object."""
        assets_path = os.path.join(os.path.dirname(__file__), "../assets")
        self_config = load_config(os.path.join(assets_path, config_path))
        self_config["datamodule"]["data_dict_path"] = os.path.join(
            assets_path, self_config["datamodule"]["data_dict_path"]
        )
        self_config["datamodule"]["split_path"] = os.path.join(assets_path, self_config["datamodule"]["split_path"])
        pika = Pika(self_config)
        pika.train()
        return pika
