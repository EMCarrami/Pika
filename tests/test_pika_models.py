from typing import cast
from unittest import TestCase

import torch
from torch import Tensor

from pika.model.pika_model import PikaModel


class TestCrossPika(TestCase):
    """Test cross attention into LLM."""

    model: PikaModel
    input_text: str
    fake_hidden: Tensor
    input_ids: Tensor
    output: str
    pad: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.pad = "<|endoftext|>"
        cls.model = PikaModel(language_model="gpt2", protein_model="esm2_t6_8M_UR50D", multimodal_strategy="cross-pika")
        cls.input_text = f"Once upon a time{cls.pad}"
        cls.fake_hidden = torch.rand(1, 50, cast(int, cls.model.esm.embed_tokens.embedding_dim))
        cls.input_ids = cls.model.text_tokenizer.encode(cls.input_text, return_tensors="pt")
        cls.output = cls._get_model_output(cls.input_ids)

    @classmethod
    def _get_model_output(self, input_ids: Tensor, max_length: int = 20) -> str:
        attention_mask = input_ids.ne(self.model.text_tokenizer.pad_token_id)
        out = self.model.pika_llm.generate(
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
        self.model.pika_llm.transformer.h[0].cross_attn.attn_gate = torch.nn.Parameter(torch.tensor([1.0]))
        self.model.pika_llm.transformer.h[0].cross_attn.ff_gate = torch.nn.Parameter(torch.tensor([1.0]))
        output_1 = self._get_model_output(self.input_ids)
        # close the gates to ensure return to default
        self.model.pika_llm.transformer.h[0].cross_attn.attn_gate = torch.nn.Parameter(torch.tensor([0.0]))
        self.model.pika_llm.transformer.h[0].cross_attn.ff_gate = torch.nn.Parameter(torch.tensor([0.0]))
        output_2 = self._get_model_output(self.input_ids)

        self.assertNotEqual(self.output, output_1)
        self.assertEqual(self.output, output_2)

    def test_pad_masking(self) -> None:
        """Test that masking pad token works."""
        new_input_ids = self.model.text_tokenizer.encode(f"{self.input_text}{self.pad}", return_tensors="pt")
        new_output = self._get_model_output(new_input_ids, max_length=21)
        self.assertEqual(new_output.replace(f"{self.pad}{self.pad}", self.pad), self.output)
