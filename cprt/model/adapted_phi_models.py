from typing import Any, Dict, List, Optional

from torch import FloatTensor, LongTensor, Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast

from .original_phi.modeling_phi import (  # type: ignore[attr-defined]
    CausalLMHead,
    CausalLMLoss,
    PhiConfig,
    PhiForCausalLM,
    PhiModel,
    PhiPreTrainedModel,
)


class PhiCPrt(PhiForCausalLM):
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
        self, *args: Any, inputs_embeds: Tensor | None = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Add encoder_hidden_states to model_inputs of GPT2 generation for CPrt."""
        model_inputs: Dict[str, Any] = super(PhiCPrt, self).prepare_inputs_for_generation(
            *args, inputs_embeds=inputs_embeds, **kwargs
        )
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
