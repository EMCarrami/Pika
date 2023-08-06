from typing import cast
from unittest import TestCase

import torch
from torch import Tensor

from cprt.model.cprt_model import Cprt


class TestCprtModel(TestCase):
    """Test cross attention into LLM."""

    model: Cprt
    input_text: str
    fake_hidden: Tensor
    input_ids: Tensor
    output: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Cprt()
        cls.input_text = "Once upon a time<PAD>"
        cls.fake_hidden = torch.rand(
            1, 50, cast(int, cls.model.esm.embed_tokens.embedding_dim)
        )
        cls.input_ids = cls.model.text_tokenizer.encode(
            cls.input_text, return_tensors="pt"
        )
        cls.output = cls.__get_model_output(cls.input_ids)

    @classmethod
    def __get_model_output(self, input_ids: Tensor, max_length: int = 20) -> str:
        attention_mask = input_ids.ne(self.model.text_tokenizer.pad_token_id)
        out = self.model.cprt_llm.generate(
            input_ids,
            max_length=max_length,
            use_cache=False,
            encoder_hidden_states=self.fake_hidden,
            attention_mask=attention_mask,
        )
        text: str = self.model.text_tokenizer.decode(out[0], skip_special_tokens=False)
        return text

    def test_info_injection(self) -> None:
        """Test information injection changes output and that it is reversible."""
        # inject information
        self.model.cprt_llm.transformer.h[0].attn_gate = torch.nn.Parameter(
            torch.tensor([1.0])
        )
        self.model.cprt_llm.transformer.h[0].ff_gate = torch.nn.Parameter(
            torch.tensor([1.0])
        )
        output_1 = self.__get_model_output(self.input_ids)
        # close the gates to ensure return to default
        self.model.cprt_llm.transformer.h[0].attn_gate = torch.nn.Parameter(
            torch.tensor([0.0])
        )
        self.model.cprt_llm.transformer.h[0].ff_gate = torch.nn.Parameter(
            torch.tensor([0.0])
        )
        output_2 = self.__get_model_output(self.input_ids)

        self.assertNotEqual(self.output, output_1)
        self.assertEqual(self.output, output_2)

    def test_pad_masking(self) -> None:
        """Test that masking <PAD> token works."""
        new_input_ids = self.model.text_tokenizer.encode(
            f"{self.input_text}<PAD>", return_tensors="pt"
        )
        new_output = self.__get_model_output(new_input_ids, max_length=21)
        self.assertEqual(new_output.replace("<PAD> <PAD>", "<PAD>"), self.output)
