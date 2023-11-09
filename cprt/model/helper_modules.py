from typing import Any, Dict, Optional, Tuple, Union, cast

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block


class GPT2CPrt(GPT2LMHeadModel):  # type: ignore[misc]
    """Modified generation config of GPT2 to allow for cross attention in generation."""

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Add encoder_hidden_states to model_inputs of GPT2 generation for CPrt."""
        model_inputs: Dict[str, Any] = super(GPT2CPrt, self).prepare_inputs_for_generation(*args, **kwargs)
        model_inputs["encoder_hidden_states"] = kwargs.get("encoder_hidden_states")
        return model_inputs


class CPrtLayer(nn.Module):
    """Perceiver and Cross attention decoder layer to inject into a default GPT decoder."""

    def __init__(
        self,
        protein_emb_dim: int,
        decoder: GPT2Block,
        num_perceiver_heads: int,
        perceiver_latent_size: int,
        num_perceiver_layers: int,
        enable_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        text_emb_dim = decoder.attn.c_attn.weight.size(0)
        dropout = decoder.attn.attn_dropout.p
        self.enable_gradient_checkpointing = enable_gradient_checkpointing

        self.protein_layer_norm = nn.LayerNorm(protein_emb_dim)
        self.perceiver = Perceiver(
            protein_emb_dim, perceiver_latent_size, text_emb_dim, num_perceiver_heads, num_perceiver_layers, dropout
        )
        self.cross_attn = GatedCrossAttentionLayer(text_emb_dim, decoder.attn.num_heads, dropout)
        self.decoder = decoder

    def forward(
        self,
        hidden_states: Tensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:

        # TODO: implement this. Need to see how best to handle it for cross attention
        #   Note to self: In generation only the last token is passed to the model, so k, v must be cached
        assert use_cache is False, "use_cache not implemented yet."
        # TODO: implement this. Note that self.config.add_cross_attention must be set True for aggregation
        assert output_attentions is False, "output_attentions for cross-attention not implemented yet."
        assert encoder_hidden_states is not None

        encoder_hidden_states = self.protein_layer_norm(encoder_hidden_states)
        if self.training and self.enable_gradient_checkpointing:
            protein_latents = checkpoint(self.perceiver, encoder_hidden_states)
            hidden_states = checkpoint(self.cross_attn, hidden_states, protein_latents)
        else:
            protein_latents = self.perceiver(encoder_hidden_states)
            hidden_states = self.cross_attn(hidden_states, protein_latents)

        return self.decoder(  # type: ignore[no-any-return]
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )


class Perceiver(nn.Module):
    """Perceiver module that handles dim mismatch."""

    def __init__(
        self, input_dim: int, latent_size: int, emb_dim: int, num_heads: int, num_layers: int, dropout: float
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
        self.latents = nn.Parameter(torch.randn(latent_size, input_dim))
        self.layers = nn.ModuleList([PerceiverLayer(input_dim, num_heads, dropout) for _ in range(num_layers)])
        self.output_proj = nn.Linear(input_dim, emb_dim, bias=False)
        self.out_layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        latents = self.latents.repeat(hidden_states.size(0), 1, 1)
        for layer in self.layers:
            latents = layer(latents, hidden_states)
        out = self.output_proj(latents)
        out: Tensor = self.out_layer_norm(out)
        return out


class PerceiverLayer(nn.Module):
    """Simple Perceiver layer."""

    def __init__(self, emb_dim: int, num_heads: int, dropout: float) -> None:
        """Init."""
        super().__init__()
        self.latent_layer_norm = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForwardNetwork(emb_dim, dropout, ff_expansion=0.5)
        self.output_layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, latents: Tensor, hidden_states: Tensor) -> Tensor:
        """Cross-attend hidden_states and latents and self-attend latents."""
        latents = self.latent_layer_norm(latents)
        residuals = latents
        hidden_latents = torch.cat((hidden_states, latents), dim=-2)
        latents, _ = self.attn(latents, hidden_latents, hidden_latents)
        latents = self.ffn(residuals + latents) + residuals
        out: Tensor = self.output_layer_norm(latents)
        return out


class GatedCrossAttentionLayer(nn.Module):
    """Gated Cross Attention."""

    def __init__(self, emb_dim: int, num_heads: int, dropout: float) -> None:
        super(GatedCrossAttentionLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.cross_attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)

        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.ffn = FeedForwardNetwork(emb_dim, dropout)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, text_embs: Tensor, protein_latents: Tensor) -> Tensor:
        """Cross attend from text to protein."""
        attn_out, _ = self.cross_attn(
            self.layer_norm(text_embs),
            protein_latents,
            protein_latents,
        )

        # TODO: consider making the gate dependant on the embedding:
        #   (chooses how much each word embedding should care about the protein sequences)
        hidden_states = attn_out * self.attn_gate + text_embs
        hidden_states = self.ffn(hidden_states)
        hidden_states: Tensor = hidden_states * self.ff_gate + text_embs
        return hidden_states


class FeedForwardNetwork(nn.Module):
    """General FFN module."""

    def __init__(self, emb_dim: int, dropout: float, ff_expansion: float = 2.0) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, int(ff_expansion * emb_dim)),
            nn.GELU(),
            nn.Linear(int(ff_expansion * emb_dim), emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        out: Tensor = self.ffn(hidden_states)
        return out


class TruncatedESM2(nn.Module):
    """Truncates an existing pre-trained ESM2 model."""

    def __init__(self, pretrained_model: nn.Module, layer_to_keep: int):
        """
        Truncate an ESM2 model to only compute the necessary encodings.

        :param pretrained_model: Initialized ESM2 pre-trained model
        :param layer_to_keep: number of transformer layers to keep (if <0 counts the layers from the end)
        """
        super().__init__()
        assert pretrained_model.__class__.__name__ == "ESM2"
        self.embed_tokens = cast(nn.Module, pretrained_model.embed_tokens)
        self.register_buffer("embed_scale", torch.tensor(pretrained_model.embed_scale))
        self.padding_idx = cast(int, pretrained_model.padding_idx)

        layers = cast(nn.ModuleList, pretrained_model.layers)
        num_layers = len(layers)
        if layer_to_keep < 0:
            total_layers = num_layers
            layer_to_keep = total_layers + layer_to_keep + 1
        assert 0 < layer_to_keep <= num_layers

        self.layers = nn.ModuleList([layers[i] for i in range(layer_to_keep)])

    def forward(self, tokens: Tensor) -> Tensor:
        """Apply ESM2 encoding operations."""
        x = self.embed_tokens(tokens) * self.embed_scale

        padding_mask = tokens.eq(self.padding_idx)
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        x = x.transpose(0, 1)
        for layer in self.layers:
            x, _attn = layer(x, self_attn_padding_mask=padding_mask)
        out: Tensor = x.transpose(0, 1)
        return out


# TODO: Implement double-cross-attention:
#   Latents query the "question" first to see what should be queried from protein.
#   Then conditioned latents query the protein, which is then used for cross attention to the text for generation.
