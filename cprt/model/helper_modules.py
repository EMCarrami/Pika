import math
from typing import Any, Dict, Literal, Optional, Tuple, Union, cast

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers import GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.phi.modeling_phi import PhiDecoderLayer


class GPT2CPrt(GPT2LMHeadModel):  # type: ignore[misc]
    """Modified generation config of GPT2 to allow for cross attention in generation."""

    def prepare_inputs_for_generation(
        self, *args: Any, inputs_embeds: Tensor | None = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Add encoder_hidden_states to model_inputs of GPT2 generation for CPrt."""
        model_inputs: Dict[str, Any] = super(GPT2CPrt, self).prepare_inputs_for_generation(
            *args, inputs_embeds=inputs_embeds, **kwargs
        )
        if kwargs.get("encoder_hidden_states", None) is not None:
            model_inputs["encoder_hidden_states"] = kwargs.get("encoder_hidden_states")
        return model_inputs


class CPrtSoftPromptLayer(nn.Module):
    """Perceiver and soft-prompt decoder to inject before the first decoder layer of LLM."""

    def __init__(
        self,
        protein_emb_dim: int,
        text_emb_dim: int,
        decoder: GPT2Block | PhiDecoderLayer,
        num_decoder_heads: int,
        num_perceiver_heads: int,
        perceiver_latent_size: int,
        num_perceiver_layers: int,
        enable_gradient_checkpointing: bool,
        dropout: float,
    ) -> None:
        super(CPrtSoftPromptLayer, self).__init__()
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.protein_layer_norm = nn.LayerNorm(protein_emb_dim)
        self.perceiver = Perceiver(
            protein_emb_dim, perceiver_latent_size, text_emb_dim, num_perceiver_heads, num_perceiver_layers, dropout
        )
        self.decoder = decoder

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        encoder_hidden_states: Tensor,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Any,
    ) -> Union[Tuple[Tensor], Optional[Tuple[Tensor, Tuple[Tensor, ...]]]]:
        """Forward with soft-prompting.

        kwargs maybe different between different models.
        e.g. past_key_value in Phi vs layer_past in GPT2
        """
        assert encoder_hidden_states is not None
        encoder_hidden_states = self.protein_layer_norm(encoder_hidden_states)
        if self.training and self.enable_gradient_checkpointing:
            protein_latents = checkpoint(self.perceiver, encoder_hidden_states)
        else:
            protein_latents = self.perceiver(encoder_hidden_states)
        return self.decoder(  # type: ignore[no-any-return]
            torch.concat([protein_latents, hidden_states[:, protein_latents.size(1) :]], dim=1),
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )


class CPrtCrossAttentionLayer(nn.Module):
    """Perceiver and Cross attention decoder layer to inject into an LLM decoder."""

    def __init__(
        self,
        protein_emb_dim: int,
        text_emb_dim: int,
        decoder: GPT2Block | PhiDecoderLayer,
        num_decoder_heads: int,
        num_perceiver_heads: int,
        perceiver_latent_size: int,
        num_perceiver_layers: int,
        enable_gradient_checkpointing: bool,
        dropout: float,
    ) -> None:
        super(CPrtCrossAttentionLayer, self).__init__()
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.protein_layer_norm = nn.LayerNorm(protein_emb_dim)
        self.perceiver = Perceiver(
            protein_emb_dim, perceiver_latent_size, text_emb_dim, num_perceiver_heads, num_perceiver_layers, dropout
        )
        self.cross_attn = GatedCrossAttentionLayer(text_emb_dim, num_decoder_heads, dropout)
        self.decoder = decoder

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        encoder_hidden_states: Tensor,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Any,
    ) -> Union[Tuple[Tensor], Optional[Tuple[Tensor, Tuple[Tensor, ...]]]]:
        """Forward with cross-attention.

        kwargs maybe different between different models.
        e.g. past_key_value in Phi vs layer_past in GPT2
        """
        # TODO: implement this. Need to see how best to handle it for cross attention
        #   Note to self: In generation only the last token is passed to the model, so k, v must be cached
        assert use_cache is False, "use_cache not implemented yet."
        # TODO: implement this. Note that self.config.add_cross_attention must be set True for aggregation
        assert output_attentions is False, "output_attentions for cross-attention not implemented yet."
        assert encoder_hidden_states is not None

        encoder_hidden_states = self.protein_layer_norm(encoder_hidden_states)
        # decoder starts with layernorm, so hidden states don't need layernorm here.
        if self.training and self.enable_gradient_checkpointing:
            protein_latents = checkpoint(self.perceiver, encoder_hidden_states)
            hidden_states = checkpoint(self.cross_attn, hidden_states, protein_latents)
        else:
            protein_latents = self.perceiver(encoder_hidden_states)
            hidden_states = self.cross_attn(hidden_states, protein_latents)
        # encoder_hidden_states will be ignored from here
        return self.decoder(  # type: ignore[no-any-return]
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )


class CPrtEncoderSkipLayer(nn.Module):
    """Module to remove encoder states when not needed."""

    def __init__(
        self,
        decoder: GPT2Block | PhiDecoderLayer,
    ) -> None:
        super(CPrtEncoderSkipLayer, self).__init__()
        self.decoder = decoder

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        encoder_hidden_states: Tensor,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Any,
    ) -> Union[Tuple[Tensor], Optional[Tuple[Tensor, Tuple[Tensor, ...]]]]:
        """Use same input as CPrtCrossAttentionLayer, just dropping the encoder_hidden_states."""
        # encoder_hidden_states will be ignored from here
        return self.decoder(  # type: ignore[no-any-return]
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )


class Perceiver(nn.Module):
    """Perceiver module that handles dim mismatch."""

    def __init__(
        self, input_dim: int, latent_size: int, output_dim: int, num_heads: int, num_layers: int, dropout: float
    ) -> None:
        """
        Initialize Perceiver.

        Takes as input an embedded sequence of any_len x input_dim and generates latent_size x emb_dim output.
        The correction of embedding dimensions of the input sequence occurs before the attention blocks.
        :param input_dim: embedding dimension of the input hidden states
        :param latent_size: length of the latent dimension
        :param output_dim: embedding dimension of the output latents
        :param num_heads: number of attention heads
        :param num_layers: number of transformer layers
        :param dropout: dropout rate
        """
        super().__init__()
        self.latents = nn.Parameter(torch.randn(latent_size, input_dim))
        self.latent_layer_norm = nn.LayerNorm(input_dim)
        self.perceiver = PerceiverLayer(input_dim, num_heads, dropout)
        self.self_attention_layers = nn.ModuleList(
            [AttentionLayer(input_dim, num_heads, dropout, ff_expansion=1) for _ in range(num_layers - 1)]
        )
        self.output_proj = nn.Linear(input_dim, output_dim, bias=False)
        self.out_layer_norm = nn.LayerNorm(output_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        latents = self.latents.repeat(hidden_states.size(0), 1, 1)
        latents = self.latent_layer_norm(latents)
        latents = self.perceiver(latents, hidden_states)
        for layer in self.self_attention_layers:
            latents = layer(latents)
        out = self.output_proj(latents)
        out: Tensor = self.out_layer_norm(out)
        return out


class PerceiverLayer(nn.Module):
    """Simple Perceiver layer."""

    def __init__(self, emb_dim: int, num_heads: int, dropout: float) -> None:
        """Init."""
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForwardNetwork(emb_dim, dropout, ff_expansion=0.5)
        self.output_layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, latents: Tensor, hidden_states: Tensor) -> Tensor:
        """Cross-attend hidden_states and latents and self-attend latents."""
        residuals = latents
        hidden_latents = torch.cat((hidden_states, latents), dim=-2)
        latents, _ = self.attn(latents, hidden_latents, hidden_latents)
        latents = self.ffn(residuals + latents) + residuals
        out: Tensor = self.output_layer_norm(latents)
        return out


class AttentionLayer(nn.Module):
    """Simple self attenition layer."""

    def __init__(self, emb_dim: int, num_heads: int, dropout: float, ff_expansion: int = 2) -> None:
        """Init."""
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = FeedForwardNetwork(emb_dim, dropout, ff_expansion=ff_expansion)
        self.output_layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """self-attend."""
        residuals = hidden_states
        hidden_states, _ = self.attn(hidden_states, hidden_states, hidden_states)
        hidden_states = self.ffn(residuals + hidden_states) + residuals
        out: Tensor = self.output_layer_norm(hidden_states)
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

    def __init__(self, pretrained_model: nn.Module, layer_to_keep: int | Literal["wte", "wpe"]):
        """
        Truncate an ESM2 model to only compute the necessary encodings.

        :param pretrained_model: Initialized ESM2 pre-trained model
        :param layer_to_keep: number of transformer layers to keep (if <0 counts the layers from the end)
        """
        super().__init__()
        assert pretrained_model.__class__.__name__ == "ESM2"
        self.padding_idx = cast(int, pretrained_model.padding_idx)
        self.embedding_dim = cast(int, pretrained_model.embed_dim)
        layers = cast(nn.ModuleList, pretrained_model.layers)
        self.num_heads = cast(int, layers[0].self_attn.num_heads)

        if isinstance(layer_to_keep, int):
            self.requires_transformers = True
            self.embed_tokens = cast(nn.Module, pretrained_model.embed_tokens)
            num_layers = len(layers)
            if layer_to_keep < 0:
                total_layers = num_layers
                layer_to_keep = total_layers + layer_to_keep + 1
            assert 0 < layer_to_keep <= num_layers
            self.layers = nn.ModuleList([layers[i] for i in range(layer_to_keep)])
        elif layer_to_keep == "wte":
            self.requires_transformers = False
            self.embed_tokens = cast(nn.Module, pretrained_model.embed_tokens)
        elif layer_to_keep == "wpe":
            self.requires_transformers = False
            text_embedder = cast(nn.Module, pretrained_model.embed_tokens)
            self.embed_tokens = nn.Sequential(text_embedder, PositionalEncoding1D(self.embedding_dim))
        else:
            raise ValueError(
                f"layer_to_keep {layer_to_keep} invalid. Should be an int indicating the transformer layer number. "
                "Alternatively, wte or wpe could be used to use aa embeddings -/+ positional embeddings, respectively."
            )

    def forward(self, tokens: Tensor) -> Tensor:
        """Apply ESM2 encoding operations."""
        x = self.embed_tokens(tokens)
        if self.requires_transformers:
            padding_mask = tokens.eq(self.padding_idx)
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))
            x = x.transpose(0, 1)
            for layer in self.layers:
                x, _attn = layer(x, self_attn_padding_mask=padding_mask)
            out: Tensor = x.transpose(0, 1)
        else:
            out = x
        return out


class PositionalEncoding1D(nn.Module):
    """Cosine positional encoding."""

    def __init__(self, emb_dim: int, max_len: int = 1500) -> None:
        super(PositionalEncoding1D, self).__init__()
        position = torch.arange(max_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe = torch.zeros(max_len, emb_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to each element in the batch."""
        x = x + self.pe[:, : x.size(1)]  # type: ignore[index]
        return x


# TODO: Implement double-cross-attention:
#   Latents query the "question" first to see what should be queried from protein.
#   Then conditioned latents query the protein, which is then used for cross attention to the text for generation.
