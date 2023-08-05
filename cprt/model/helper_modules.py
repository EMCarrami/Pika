import torch
from torch import nn, Tensor
from typing import Tuple, Optional, Union


class TruncatedESM2(nn.Module):
    """Truncates an existing pre-trained ESM2 model."""

    def __init__(self, pretrained_model: nn.Module, layer_to_keep: int):
        """
        Truncate an ESM2 model to only compute the necessary encodings.

        :param pretrained_model: Initialized ESM2 pre-trained model
        :param layer_to_keep: number of transformer layers to keep (if <0 counts the layers from the end)
        """
        super().__init__()
        assert pretrained_model.__class__.__name__ == 'ESM2'
        self.embed_tokens = pretrained_model.embed_tokens
        self.embed_scale = pretrained_model.embed_scale
        self.padding_idx = pretrained_model.padding_idx

        if layer_to_keep < 0:
            total_layers = len(pretrained_model.layers)
            layer_to_keep = total_layers + layer_to_keep + 1
        assert 0 < layer_to_keep <= len(pretrained_model.layers)

        self.layers = nn.ModuleList([pretrained_model.layers[i] for i in range(layer_to_keep)])

    def forward(self, tokens: Tensor) -> Tensor:
        """Apply ESM2 encoding operations."""
        x = self.embed_tokens(tokens) * self.embed_scale
        padding_mask = tokens.eq(self.padding_idx)
        if padding_mask is not None:
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
        if not padding_mask.any():
            padding_mask = None
        x = x.transpose(0, 1)
        for layer in self.layers:
            x, _attn = layer(x, self_attn_padding_mask=padding_mask)
        return x.transpose(0, 1)


class FeedForwardNetwork(nn.Module):
    """General FFN module."""
    def __init__(self, emb_dim: int, dropout: float, ff_expansion: int = 4) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, ff_expansion * emb_dim),
            nn.GELU(),
            nn.Linear(ff_expansion * emb_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.ffn(hidden_states)


class PerceiverLayer(nn.Module):
    """Simple Perceiver layer."""
    def __init__(self, emb_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.latent_layer_norm = nn.LayerNorm(emb_dim)
        self.input_layer_norm = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForwardNetwork(emb_dim, dropout)

    def forward(self, latents: Tensor, hidden_states: Tensor) -> Tensor:
        """Cross-attend hidden_states and latents and self-attend latents."""
        residuals = latents
        latents = self.latent_layer_norm(latents)
        hidden_latents = torch.cat((self.input_layer_norm(hidden_states), latents), dim=-2)
        latents, _ = self.attn(latents, hidden_latents, hidden_latents)
        return self.ffn(residuals + latents) + residuals


class Perceiver(nn.Module):
    """Perceiver module that handles dim mismatch."""
    def __init__(
            self,
            input_dim: int,
            latent_size: int,
            emb_dim: int, num_heads: int, num_layers: int, dropout: float
    ) -> None:
        """
        Initialize Perceiver.

        Takes as input an embedded sequence of any_len x input_dim and generates latent_size x emb_dim output.
        The correction of embedding dimensions of the input sequence occurs before the attention blocks.
        :param input_dim: embedding dimension of the input hidden states
        :param latent_size: length of the latent dimension
        :param emb_dim: embedding dimension of the output latents
        :param num_heads: number of attention heads
        :param num_layers: number of transformer layers
        :param dropout: dropout rate
        """
        super().__init__()
        self.latents = nn.Parameter(torch.randn(latent_size, emb_dim))
        self.input_layer_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, emb_dim, bias=False)
        self.layers = nn.ModuleList(
            [PerceiverLayer(emb_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.out_layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        latents = self.latents.repeat(hidden_states.size(0), 1, 1)
        hidden_states = self.input_layer_norm(hidden_states)
        hidden_states = self.input_proj(hidden_states)
        for layer in self.layers:
            latents = layer(latents, hidden_states)
        return self.out_layer_norm(latents)


class CrossAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        protein_emb_dim: int,
        decoder: nn.Module,
        perceiver_latent_size: int = 100,
        num_perceiver_layers: int = 1
    ) -> None:
        super().__init__()
        text_emb_dim = decoder.attn.c_attn.weight.size(0)
        num_heads = decoder.attn.num_heads
        dropout = decoder.attn.attn_dropout.p

        self.perceiver = Perceiver(
            protein_emb_dim,
            perceiver_latent_size,
            text_emb_dim,
            num_heads,
            num_perceiver_layers,
            dropout
        )

        self.text_layer_norm = nn.LayerNorm(text_emb_dim)
        self.cross_attn = nn.MultiheadAttention(
            text_emb_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.ffn = FeedForwardNetwork(text_emb_dim, dropout)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))
        self.decoder = decoder

    def forward(
        self,
        hidden_states: Tensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:

        # TODO: implement this. Need to see how best to handle it for cross attention
        #   Note to self: In generation only the last token is passed to the model, so k, v must be cached
        assert use_cache is False, "use_cache not implemented yet"

        # TODO: implement this. Note that self.config.add_cross_attention must be set True for aggregation
        assert output_attentions is False, "output_attentions for cross-attention not implemented yet."

        text = hidden_states
        protein = encoder_hidden_states
        assert protein is not None
        protein = self.perceiver(protein)

        if encoder_attention_mask is not None:
            # TODO:
            #   remove the parts of the text that are not needed to attend to
            #   and only update the relevant sections in the hidden states
            pass
        attn_out, _ = self.cross_attn(
            self.text_layer_norm(text),
            protein,
            protein,
            attn_mask=encoder_attention_mask
        )

        # TODO: consider making the gate dependent on the embedding:
        #   (chooses how much each word embedding should care about the protein sequences)
        hidden_states = attn_out * self.attn_gate + text
        hidden_states = self.ffn(hidden_states)
        hidden_states = hidden_states * self.attn_gate + text

        return self.decoder(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions
        )

# TODO: Implement double-cross-attention:
#   Latents query the "question" first to see what should be queried from protein.
#   Then conditioned latents query the protein, which is then used for cross attention to the text for generation.