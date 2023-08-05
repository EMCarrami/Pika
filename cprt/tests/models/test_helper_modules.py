from unittest import TestCase
from cprt.model.helper_modules import (
    FeedForwardNetwork, Perceiver, PerceiverLayer, TruncatedESM2, CrossAttentionDecoderLayer
)
from transformers import AutoTokenizer, GPT2LMHeadModel
import unittest
import torch


class TestTruncatedTransformer(TestCase):
    def setUp(self):
        self.pretrained_model, self.alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t6_8M_UR50D")
        self.protein_tokenizer = AutoTokenizer.from_pretrained(f"facebook/esm2_t6_8M_UR50D")
        self.layer_to_keep = 2
        self.truncated_model = TruncatedESM2(self.pretrained_model, self.layer_to_keep)
        self.input_ids = self.protein_tokenizer(
            ['MAGTGASC', 'HCCM'],
            return_tensors='pt',
            padding=True,
            truncation=True
        )['input_ids']
        self.pretrained_model.eval()
        self.pretrained_model.token_dropout = False
        self.truncated_model.eval()


    def test_esm_compatibility(self):
        self.assertDictEqual(self.alphabet.to_dict(), self.protein_tokenizer.get_vocab())

    def test_truncated_transformer(self):
        truncated_output = self.truncated_model(self.input_ids)

        original_output = self.pretrained_model(
            self.input_ids, repr_layers=[self.layer_to_keep]
        )['representations'][self.layer_to_keep]

        self.assertEqual(truncated_output.shape, original_output.shape)
        self.assertTrue(torch.allclose(truncated_output, original_output))


class TestPerceiverComponents(TestCase):
    def setUp(self):
        self.emb_dim = 32
        self.input_dim = self.emb_dim + 2
        self.num_heads = 2
        self.dropout = 0.1
        self.ff_expansion = 2
        self.latent_size = 4
        self.num_layers = 2
        self.batch_size = 8
        self.seq_len = 10

    def test_feed_forward_network(self):
        ffn = FeedForwardNetwork(self.emb_dim, self.dropout, self.ff_expansion)
        x = torch.rand(self.batch_size, self.seq_len, self.emb_dim)
        out = ffn(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.emb_dim))

    def test_perceiver_layer(self):
        layer = PerceiverLayer(self.emb_dim, self.num_heads, self.dropout)
        latents = torch.rand(self.batch_size, self.latent_size, self.emb_dim)
        hidden_states = torch.rand(self.batch_size, self.seq_len, self.emb_dim)
        out = layer(latents, hidden_states)
        self.assertEqual(out.shape, (self.batch_size, self.latent_size, self.emb_dim))

    def test_perceiver(self):
        model = Perceiver(
            self.input_dim,
            self.latent_size,
            self.emb_dim,
            self.num_heads,
            self.num_layers,
            self.dropout
        )
        x = torch.rand(self.batch_size, self.seq_len, self.input_dim)
        out = model(x)
        self.assertEqual(out.shape, (self.batch_size, self.latent_size, self.emb_dim))


class TestCrossAttentionDecoderLayer(TestCase):

    def setUp(self) -> None:
        llm = GPT2LMHeadModel.from_pretrained('gpt2')
        self.decoder = llm.transformer.h[0]
        self.layer = CrossAttentionDecoderLayer(
            protein_emb_dim=10,
            decoder=self.decoder,
            perceiver_latent_size=5
        )
        self.text_emb = torch.rand(2, 50, 768)
        self.protein_emb = torch.rand(2, 8, 10)

    def test_output_shape(self) -> None:
        decoder_out = self.decoder(self.text_emb)
        cprt_layer_out = self.layer(
            self.text_emb,
            encoder_hidden_states=self.protein_emb,
            encoder_attention_mask=None
        )
        self.assertEqual(decoder_out[0].shape, cprt_layer_out[0].shape)

    def test_input_assertion(self) -> None:
        with self.assertRaises(AssertionError):
            self.layer(
                self.text_emb,
                encoder_hidden_states=None,
                encoder_attention_mask=None
            )


if __name__ == '__main__':
    unittest.main()
