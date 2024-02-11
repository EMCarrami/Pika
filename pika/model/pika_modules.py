from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import FloatTensor, LongTensor, Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers import GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.phi.modeling_phi import PhiDecoderLayer

from pika.model.helper_modules import GatedCrossAttentionLayer, Perceiver
from pika.model.original_phi.configuration_phi import PhiConfig
from pika.model.original_phi.modeling_phi import (
    CausalLMHead,
    CausalLMLoss,
    PhiForCausalLM,
    PhiModel,
    PhiPreTrainedModel,
)


class GPT2ForPika(GPT2LMHeadModel):  # type: ignore[misc]
    """Modified generation config of GPT2 to allow for cross attention in generation."""

    def prepare_inputs_for_generation(
        self, *args: Any, inputs_embeds: Tensor | None = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Add encoder_hidden_states to model_inputs of GPT2 generation for Pika."""
        model_inputs: Dict[str, Any] = super(GPT2ForPika, self).prepare_inputs_for_generation(
            *args, inputs_embeds=inputs_embeds, **kwargs
        )
        if kwargs.get("encoder_hidden_states", None) is not None:
            model_inputs["encoder_hidden_states"] = kwargs.get("encoder_hidden_states")
        return model_inputs


class PhiForPika(PhiForCausalLM):
    """Modified PhiForCausalLM to allow for cross attention in forward and generation."""

    def __init__(self, config: PhiConfig) -> None:
        # bypassing the call to PhiForCausalLM.__init__() by calling its parent's __init__() directly.
        PhiPreTrainedModel.__init__(self, config)
        self.transformer = PhiModelWithEncoderInput(config)
        self.lm_head = CausalLMHead(config)
        self.loss = CausalLMLoss()
        self.post_init()

    def forward(  # type: ignore[override]
        self,
        input_ids: LongTensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[LongTensor] = None,
        past_key_values: Optional[List[FloatTensor]] = None,
        inputs_embeds: Optional[FloatTensor] = None,
        labels: Optional[LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        encoder_hidden_states: Optional[Tensor] = None,
    ) -> CausalLMOutputWithPast:
        """
        Forward using encoder_hidden states.

        Modified to add encoder_hidden_state to the forward and pass it to the self.model.
        Everything else remains exactly the same.
        """
        hidden_states = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss(lm_logits, labels)

        return CausalLMOutputWithPast(loss=loss, logits=lm_logits, past_key_values=past_key_values)

    def prepare_inputs_for_generation(
        self,
        input_ids: Tensor,
        past_key_values: Tensor | None = None,  # type: ignore[override]
        attention_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Add encoder_hidden_states to model_inputs of Phi generation for Pika."""
        # TODO: fix generation optimization with past_key_values
        model_inputs: Dict[str, Any] = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
        }
        if kwargs.get("encoder_hidden_states", None) is not None:
            model_inputs["encoder_hidden_states"] = kwargs.get("encoder_hidden_states")
        return model_inputs


class PhiModelWithEncoderInput(PhiModel):
    """Modification to the forward of PhiModel to allow for encoder_hidden_state input."""

    def forward(  # type: ignore[override]
        self,
        input_ids: LongTensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[LongTensor] = None,
        past_key_values: Optional[List[FloatTensor]] = None,
        inputs_embeds: Optional[FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        encoder_hidden_states: Optional[Tensor] = None,
    ) -> Tensor:
        hidden_states = self.embd(input_ids)

        for layer in self.h:
            hidden_states = layer(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )
        assert isinstance(hidden_states, Tensor)
        return hidden_states


class SelfPikaLayer(nn.Module):
    """Perceiver and protein latents to inject before the first decoder layer of LLM."""

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
        super(SelfPikaLayer, self).__init__()
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
        """
        Forward with protein latents.

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


class CrossPikaAttentionLayer(nn.Module):
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
        super(CrossPikaAttentionLayer, self).__init__()
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
        """
        Forward with cross-attention.

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
            hidden_states = checkpoint(self.cross_attn, hidden_states, protein_latents, attention_mask)
        else:
            protein_latents = self.perceiver(encoder_hidden_states)
            hidden_states = self.cross_attn(hidden_states, protein_latents, attention_mask)
        # encoder_hidden_states will be ignored from here
        return self.decoder(  # type: ignore[no-any-return]
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )


class PikaEncoderSkipLayer(nn.Module):
    """Module to remove encoder states when not needed."""

    def __init__(
        self,
        decoder: GPT2Block | PhiDecoderLayer,
    ) -> None:
        super(PikaEncoderSkipLayer, self).__init__()
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
        """Use same input as CrossPikaAttentionLayer, just dropping the encoder_hidden_states."""
        # encoder_hidden_states will be ignored from here
        return self.decoder(  # type: ignore[no-any-return]
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
