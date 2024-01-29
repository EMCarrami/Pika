import os
import pickle
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from cprt.utils.chatgpt_processor import GPTProcessor  # Import your GPTProcessor class


class TestGPTProcessor(unittest.TestCase):
    """Test class for GPTProcessor's multi-threading."""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_api_key"})
    @patch("cprt.utils.gpt_processor.GPTProcessor.get_response")
    def test_bulk_process_success(self, mock_get_response: MagicMock) -> None:
        """Check outputs are correct."""
        # Mock successful responses from get_response
        mock_get_response.side_effect = lambda message, request_id: {"text": f"Response for {request_id}"}

        processor = GPTProcessor(model="text-davinci-003", secondary_model=None)

        message_list = [
            [{"role": "system", "content": "Test 1"}, {"role": "user", "content": "Hello!"}],
            [{"role": "system", "content": "Test 2"}, {"role": "user", "content": "Hello!"}],
        ]
        request_names = ["req1", "req2"]

        # Test with dict return
        response_dict = processor.bulk_process(
            message_list, request_names, num_workers=2, return_dict=True, save_dir=None
        )
        assert response_dict is not None
        self.assertEqual(response_dict["req1"], {"text": "Response for req1_0"})
        self.assertEqual(response_dict["req2"], {"text": "Response for req2_1"})

        # Test with file save
        processor.bulk_process(message_list, request_names, num_workers=2, return_dict=False, save_dir="test_dir")
        with open("test_dir/req1.pkl", "rb") as f:
            saved_response1 = pickle.load(f)
        self.assertEqual(saved_response1, {"text": "Response for req1_0"})
        with self.assertRaises(Exception):
            processor.bulk_process(message_list, request_names, num_workers=2, return_dict=False, save_dir="test_dir")

        # Cleanup
        os.remove("test_dir/req1.pkl")
        os.remove("test_dir/req2.pkl")
        os.rmdir("test_dir")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "fake_api_key"})
    @patch("cprt.utils.gpt_processor.GPTProcessor.get_response")
    def test_bulk_process_failure_stops_all(self, mock_get_response: MagicMock) -> None:
        """Check processes fail on time."""
        # Mock a failure on the second call to get_response
        def side_effect(message: Dict[str, List[Dict[str, str]]], request_id: str) -> Dict[str, Any]:
            """Raise exception for req2."""
            if "req2" in request_id:
                raise Exception("Test exception")
            return {"text": f"Response for {request_id}"}

        mock_get_response.side_effect = side_effect

        processor = GPTProcessor(model="text-davinci-003", secondary_model=None)

        # Create 100 messages, where the second one is designed to fail
        message_list = [[{"role": "system", "content": f"Test {i}"}] for i in range(100)]
        request_names = [f"req{i}" for i in range(100)]

        with self.assertRaises(Exception):
            processor.bulk_process(message_list, request_names, num_workers=2, return_dict=True, save_dir=None)

        # Check not all tasks were processed before the exception was raised
        self.assertTrue(mock_get_response.call_count < 10, "Not all tasks should have been processed")
